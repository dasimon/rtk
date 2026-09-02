//! All-tools token attribution: extracts every tool_use/tool_result pair
//! from Claude Code session JSONL (the Bash-only provider misses ~65% of bytes).

use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug)]
#[allow(dead_code)] // wired by Task 2/3 (attribution & report)
pub enum ToolEventKind {
    /// Un tool_result apparié à son tool_use.
    Result {
        tool: String,
        bytes: usize,
        /// Bash uniquement : la commande émise (avant réécriture hook).
        command: Option<String>,
        /// chemin normalisé si l'outil en fournit un (Read notamment).
        file_path: Option<String>,
    },
    /// Un tool_use Edit/Write/NotebookEdit (compté à l'émission, pas au résultat).
    FileEdit { file_path: String },
}

#[derive(Debug)]
#[allow(dead_code)] // wired by Task 2/3 (attribution & report)
pub struct ToolEvent {
    pub kind: ToolEventKind,
    pub seq: usize,
}

/// Minuscules + séparateurs unifiés : les transcripts Windows mélangent les deux.
#[allow(dead_code)] // wired by Task 2/3 (attribution & report)
pub fn normalize_path(p: &str) -> String {
    p.replace('\\', "/").to_lowercase()
}

#[allow(dead_code)] // wired by Task 2/3 (attribution & report)
const EDIT_TOOLS: &[&str] = &["Edit", "Write", "NotebookEdit"];

/// Extrait tous les événements outils d'une session, dans l'ordre du fichier.
/// Retourne (événements, nb de tool_result réellement orphelins — un tool_result
/// dont le tool_use est un Edit/Write/NotebookEdit est consommé silencieusement,
/// pas compté, puisque son événement a déjà été émis au tool_use). Ligne invalide
/// = ignorée.
#[allow(dead_code)] // wired by Task 2/3 (attribution & report)
pub fn extract_tool_events(path: &Path) -> Result<(Vec<ToolEvent>, usize)> {
    let file =
        std::fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = BufReader::new(file);

    // id -> (tool name, command Bash éventuelle, file_path Read éventuel)
    let mut pending: HashMap<String, (String, Option<String>, Option<String>)> = HashMap::new();
    // ids des tool_use Edit/Write/NotebookEdit : leur tool_result est attendu
    // (déjà compté via l'événement FileEdit) et doit être consommé sans
    // incrémenter unmatched.
    let mut edit_ids: HashSet<String> = HashSet::new();
    let mut events = Vec::new();
    let mut unmatched = 0usize;
    let mut seq = 0usize;

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        // Pré-filtre : "tool_use\"" matche le type des blocs use mais pas "tool_use_id".
        if !line.contains("\"tool_use\"") && !line.contains("\"tool_result\"") {
            continue;
        }
        let entry: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let Some(content) = entry.pointer("/message/content").and_then(|c| c.as_array()) else {
            continue;
        };
        for block in content {
            match block.get("type").and_then(|t| t.as_str()) {
                Some("tool_use") => {
                    let Some(name) = block.get("name").and_then(|n| n.as_str()) else {
                        continue;
                    };
                    let input_path = block
                        .pointer("/input/file_path")
                        .or_else(|| block.pointer("/input/notebook_path"))
                        .and_then(|p| p.as_str())
                        .map(normalize_path);
                    if EDIT_TOOLS.contains(&name) {
                        if let Some(id) = block.get("id").and_then(|i| i.as_str()) {
                            edit_ids.insert(id.to_string());
                        }
                        if let Some(p) = input_path {
                            events.push(ToolEvent {
                                kind: ToolEventKind::FileEdit { file_path: p },
                                seq,
                            });
                            seq += 1;
                        }
                        continue;
                    }
                    if let Some(id) = block.get("id").and_then(|i| i.as_str()) {
                        let command = block
                            .pointer("/input/command")
                            .and_then(|c| c.as_str())
                            .filter(|_| name == "Bash")
                            .map(str::to_string);
                        pending.insert(id.to_string(), (name.to_string(), command, input_path));
                    }
                }
                Some("tool_result") => {
                    let Some(id) = block.get("tool_use_id").and_then(|i| i.as_str()) else {
                        continue;
                    };
                    if edit_ids.remove(id) {
                        // Résultat d'un Edit/Write/NotebookEdit : déjà compté au
                        // tool_use, consommé sans incrémenter unmatched.
                        continue;
                    }
                    let Some((tool, command, file_path)) = pending.remove(id) else {
                        unmatched += 1;
                        continue;
                    };
                    let bytes = match block.get("content") {
                        Some(serde_json::Value::String(s)) => s.len(),
                        Some(other) => serde_json::to_string(other).map(|s| s.len()).unwrap_or(0),
                        None => 0,
                    };
                    events.push(ToolEvent {
                        kind: ToolEventKind::Result {
                            tool,
                            bytes,
                            command,
                            file_path,
                        },
                        seq,
                    });
                    seq += 1;
                }
                _ => {}
            }
        }
    }
    Ok((events, unmatched))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_jsonl(lines: &[&str]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        for l in lines {
            writeln!(f, "{}", l).unwrap();
        }
        f
    }

    const READ_USE: &str = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Read","input":{"file_path":"C:\\Proj\\A.rs"}}]}}"#;
    const READ_RESULT: &str = r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t1","content":"hello"}]}}"#;

    #[test]
    fn test_extract_read_pair() {
        let f = make_jsonl(&[READ_USE, READ_RESULT]);
        let (events, unmatched) = extract_tool_events(f.path()).unwrap();
        assert_eq!(unmatched, 0);
        assert_eq!(events.len(), 1);
        match &events[0].kind {
            ToolEventKind::Result {
                tool,
                bytes,
                file_path,
                command,
            } => {
                assert_eq!(tool, "Read");
                assert_eq!(*bytes, 5); // "hello"
                assert_eq!(file_path.as_deref(), Some("c:/proj/a.rs")); // normalisé
                assert!(command.is_none());
            }
            _ => panic!("expected Result"),
        }
    }

    #[test]
    fn test_extract_array_content_uses_json_len() {
        let use_l = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t2","name":"mcp__grafana__query_prometheus","input":{}}]}}"#;
        let res_l = r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t2","content":[{"type":"text","text":"xy"}]}]}}"#;
        let f = make_jsonl(&[use_l, res_l]);
        let (events, _) = extract_tool_events(f.path()).unwrap();
        match &events[0].kind {
            ToolEventKind::Result { tool, bytes, .. } => {
                assert_eq!(tool, "mcp__grafana__query_prometheus");
                // longueur de serde_json::to_string du tableau content
                assert_eq!(*bytes, r#"[{"type":"text","text":"xy"}]"#.len());
            }
            _ => panic!("expected Result"),
        }
    }

    #[test]
    fn test_extract_edit_event_and_orphan_result() {
        let edit_l = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t3","name":"Edit","input":{"file_path":"C:\\Proj\\A.rs","old_string":"x","new_string":"y"}}]}}"#;
        let orphan = r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"zz","content":"lost"}]}}"#;
        let f = make_jsonl(&[edit_l, orphan, "not json at all"]);
        let (events, unmatched) = extract_tool_events(f.path()).unwrap();
        assert_eq!(unmatched, 1);
        assert_eq!(events.len(), 1);
        match &events[0].kind {
            ToolEventKind::FileEdit { file_path } => assert_eq!(file_path, "c:/proj/a.rs"),
            _ => panic!("expected FileEdit"),
        }
    }

    #[test]
    fn test_extract_notebookedit_uses_notebook_path() {
        let nb_l = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"n1","name":"NotebookEdit","input":{"notebook_path":"C:\\proj\\nb.ipynb","new_source":"x"}}]}}"#;
        let f = make_jsonl(&[nb_l]);
        let (events, unmatched) = extract_tool_events(f.path()).unwrap();
        assert_eq!(unmatched, 0);
        assert_eq!(events.len(), 1);
        match &events[0].kind {
            ToolEventKind::FileEdit { file_path } => assert_eq!(file_path, "c:/proj/nb.ipynb"),
            _ => panic!("expected FileEdit"),
        }
    }

    #[test]
    fn test_extract_edit_result_consumed_not_unmatched() {
        let edit_l = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"e1","name":"Edit","input":{"file_path":"C:\\Proj\\A.rs","old_string":"x","new_string":"y"}}]}}"#;
        let edit_res = r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"e1","content":"ignored"}]}}"#;
        let f = make_jsonl(&[edit_l, edit_res]);
        let (events, unmatched) = extract_tool_events(f.path()).unwrap();
        assert_eq!(unmatched, 0);
        assert_eq!(events.len(), 1);
        match &events[0].kind {
            ToolEventKind::FileEdit { file_path } => assert_eq!(file_path, "c:/proj/a.rs"),
            _ => panic!("expected FileEdit"),
        }
    }

    #[test]
    fn test_extract_bash_carries_command_and_seq_order() {
        let bash_use = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t4","name":"Bash","input":{"command":"rtk git status"}}]}}"#;
        let bash_res = r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t4","content":"ok"}]}}"#;
        let edit_l = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t5","name":"Write","input":{"file_path":"b.txt","content":"z"}}]}}"#;
        let f = make_jsonl(&[bash_use, bash_res, edit_l]);
        let (events, _) = extract_tool_events(f.path()).unwrap();
        assert_eq!(events.len(), 2);
        // le Bash result arrive avant le Write use dans le fichier -> seq croissants
        assert!(events[0].seq < events[1].seq);
        match &events[0].kind {
            ToolEventKind::Result { command, .. } => {
                assert_eq!(command.as_deref(), Some("rtk git status"))
            }
            _ => panic!("expected Result"),
        }
    }

    #[test]
    fn test_normalize_path() {
        assert_eq!(normalize_path("C:\\Users\\X\\F.RS"), "c:/users/x/f.rs");
    }
}
