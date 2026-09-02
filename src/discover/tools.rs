//! All-tools token attribution: extracts every tool_use/tool_result pair
//! from Claude Code session JSONL (the Bash-only provider misses ~65% of bytes).

use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug)]
#[allow(dead_code)] // wired by Task 5 (CLI)
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
#[allow(dead_code)] // wired by Task 5 (CLI)
pub struct ToolEvent {
    pub kind: ToolEventKind,
    pub seq: usize,
}

/// Minuscules + séparateurs unifiés : les transcripts Windows mélangent les deux.
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

/// Extensions dont le contenu lu est binaire/extrait : incompressible par un filtre.
#[allow(dead_code)] // wired by Task 5 (CLI)
const BINARY_EXTS: &[&str] = &[
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".xls", ".xlsx", ".doc", ".docx",
    ".ppt", ".pptx", ".zip",
];

#[allow(dead_code)] // wired by Task 5 (CLI)
/// Attend un chemin déjà normalisé par `normalize_path` (minuscules, `/`).
fn is_binary_path(p: &str) -> bool {
    BINARY_EXTS.iter().any(|e| p.ends_with(e))
}

#[derive(Debug, Default, serde::Serialize)]
pub struct ReadBreakdown {
    pub total_calls: usize,
    pub total_bytes: usize,
    pub binary_bytes: usize,
    pub pure_reread_calls: usize,
    pub pure_reread_bytes: usize,
    pub postedit_reread_calls: usize,
    pub postedit_reread_bytes: usize,
    pub locked_bytes: usize,
    pub cross_session_first_bytes: usize,
}

#[derive(Debug, Default)]
#[allow(dead_code)] // wired by Task 3 (report)
struct OffenderStat {
    bytes: usize,
    calls: usize,
    max_calls_one_session: usize,
    edited: bool,
    binary: bool,
}

#[allow(dead_code)] // wired by Task 3/5 (report & CLI)
pub struct AllToolsAnalyzer {
    families: HashMap<String, (usize, usize)>, // family -> (calls, bytes)
    read: ReadBreakdown,
    offenders: HashMap<String, OffenderStat>,
    /// Combien de sessions ANTÉRIEURES ont lu ce chemin.
    seen_paths: HashMap<String, usize>,
    unmatched_results: usize,
    rtk_re: regex::Regex,
}

impl Default for AllToolsAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)] // wired by Task 3/5 (report & CLI)
impl AllToolsAnalyzer {
    pub fn new() -> Self {
        Self {
            families: HashMap::new(),
            read: ReadBreakdown::default(),
            offenders: HashMap::new(),
            seen_paths: HashMap::new(),
            unmatched_results: 0,
            rtk_re: regex::Regex::new(r"(^|[\s;|&(])rtk\s").expect("static regex"),
        }
    }

    #[doc(hidden)]
    pub fn read_breakdown(&self) -> &ReadBreakdown {
        &self.read
    }

    fn family_key(&self, tool: &str, command: Option<&str>) -> String {
        if tool == "Bash" {
            let rtk = command.map(|c| self.rtk_re.is_match(c)).unwrap_or(false);
            return if rtk {
                "Bash (rtk)".into()
            } else {
                "Bash (raw)".into()
            };
        }
        if let Some(rest) = tool.strip_prefix("mcp__") {
            let server = rest.split("__").next().unwrap_or(rest);
            return format!("MCP:{}", server);
        }
        if tool == "Task" || tool == "Agent" {
            return "Task/Agent".into();
        }
        tool.to_string()
    }

    pub fn ingest_session(&mut self, events: &[ToolEvent], unmatched: usize) {
        self.unmatched_results += unmatched;
        // Passe 1 : positions des éditions par chemin.
        let mut edit_seqs: HashMap<&str, Vec<usize>> = HashMap::new();
        for e in events {
            if let ToolEventKind::FileEdit { file_path } = &e.kind {
                edit_seqs.entry(file_path.as_str()).or_default().push(e.seq);
            }
        }
        // Passe 2 : résultats.
        let mut last_read_seq: HashMap<&str, usize> = HashMap::new();
        let mut session_reads: HashMap<&str, usize> = HashMap::new();
        for e in events {
            let ToolEventKind::Result {
                tool,
                bytes,
                command,
                file_path,
            } = &e.kind
            else {
                continue;
            };
            let fam = self.family_key(tool, command.as_deref());
            let entry = self.families.entry(fam).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += bytes;

            let offender_key: Option<String> = if tool == "Read" {
                file_path.clone()
            } else if tool.starts_with("mcp__") {
                Some(self.family_key(tool, None))
            } else {
                None
            };
            if let Some(key) = &offender_key {
                let o = self.offenders.entry(key.clone()).or_default();
                o.bytes += bytes;
                o.calls += 1;
            }

            if tool != "Read" {
                continue;
            }
            let Some(p) = file_path.as_deref() else {
                continue;
            };
            self.read.total_calls += 1;
            self.read.total_bytes += bytes;
            let binary = is_binary_path(p);
            if binary {
                self.read.binary_bytes += bytes;
            }
            let edits = edit_seqs.get(p).map(Vec::as_slice).unwrap_or(&[]);
            let locked = edits.iter().any(|&s| s > e.seq);
            if locked {
                self.read.locked_bytes += bytes;
            }
            if let Some(&prev) = last_read_seq.get(p) {
                let inter_edit = edits.iter().any(|&s| s > prev && s < e.seq);
                if inter_edit {
                    self.read.postedit_reread_calls += 1;
                    self.read.postedit_reread_bytes += bytes;
                } else {
                    self.read.pure_reread_calls += 1;
                    self.read.pure_reread_bytes += bytes;
                }
            } else if self.seen_paths.get(p).copied().unwrap_or(0) > 0 {
                self.read.cross_session_first_bytes += bytes;
            }
            if let Some(key) = &offender_key {
                let o = self.offenders.entry(key.clone()).or_default();
                o.edited |= !edits.is_empty();
                o.binary |= binary;
            }
            *session_reads.entry(p).or_insert(0) += 1;
            last_read_seq.insert(p, e.seq);
        }
        // Clôture session : cross-session + max par session.
        for (p, n) in session_reads {
            *self.seen_paths.entry(p.to_string()).or_insert(0) += 1;
            if let Some(o) = self.offenders.get_mut(p) {
                o.max_calls_one_session = o.max_calls_one_session.max(n);
            }
        }
    }

    /// Consomme l'analyzer et produit le rapport agrégé : familles triées par
    /// octets décroissants, top `limit` offenders (idem), totaux.
    pub fn finish(self, limit: usize) -> AllToolsReport {
        let mut families: Vec<FamilyStat> = self
            .families
            .into_iter()
            .map(|(family, (calls, bytes))| FamilyStat {
                family,
                calls,
                bytes,
            })
            .collect();
        families.sort_by(|a, b| b.bytes.cmp(&a.bytes).then(a.family.cmp(&b.family)));
        let total_bytes = families.iter().map(|f| f.bytes).sum();
        let mut offenders: Vec<OffenderEntry> = self
            .offenders
            .into_iter()
            .map(|(key, o)| OffenderEntry {
                key,
                bytes: o.bytes,
                calls: o.calls,
                max_calls_one_session: o.max_calls_one_session,
                edited: o.edited,
                binary: o.binary,
            })
            .collect();
        offenders.sort_by(|a, b| b.bytes.cmp(&a.bytes).then(a.key.cmp(&b.key)));
        offenders.truncate(limit);
        AllToolsReport {
            families,
            read: self.read,
            offenders,
            unmatched_results: self.unmatched_results,
            total_bytes,
        }
    }
}

#[derive(Debug, serde::Serialize)]
#[allow(dead_code)] // wired by Task 4/5 (rendu & CLI)
pub struct FamilyStat {
    pub family: String,
    pub calls: usize,
    pub bytes: usize,
}

#[derive(Debug, serde::Serialize)]
#[allow(dead_code)] // wired by Task 4/5 (rendu & CLI)
pub struct OffenderEntry {
    pub key: String,
    pub bytes: usize,
    pub calls: usize,
    pub max_calls_one_session: usize,
    pub edited: bool,
    pub binary: bool,
}

#[derive(Debug, serde::Serialize)]
#[allow(dead_code)] // wired by Task 4/5 (rendu & CLI)
pub struct AllToolsReport {
    pub families: Vec<FamilyStat>,
    /// Détail des Read (voir `ReadBreakdown`).
    /// ⚠️ Écart de réconciliation connu : un tool_result Read sans `file_path`
    /// (ex. Read refusé/erreur avant résolution du chemin) est compté dans
    /// `families["Read"]` (calls/bytes) mais PAS dans ce breakdown, qui exige
    /// un chemin pour classer la lecture (reread, locked, binaire, etc.).
    /// `read.total_bytes` peut donc être strictement inférieur aux octets de
    /// la famille "Read".
    pub read: ReadBreakdown,
    pub offenders: Vec<OffenderEntry>,
    pub unmatched_results: usize,
    pub total_bytes: usize,
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

    fn read_ev(seq: usize, path: &str, bytes: usize) -> ToolEvent {
        ToolEvent {
            kind: ToolEventKind::Result {
                tool: "Read".into(),
                bytes,
                command: None,
                file_path: Some(normalize_path(path)),
            },
            seq,
        }
    }
    fn edit_ev(seq: usize, path: &str) -> ToolEvent {
        ToolEvent {
            kind: ToolEventKind::FileEdit {
                file_path: normalize_path(path),
            },
            seq,
        }
    }

    #[test]
    fn test_pure_reread_same_session() {
        let mut a = AllToolsAnalyzer::new();
        a.ingest_session(&[read_ev(0, "a.rs", 100), read_ev(1, "a.rs", 100)], 0);
        let r = a.read_breakdown();
        assert_eq!(r.pure_reread_calls, 1);
        assert_eq!(r.pure_reread_bytes, 100); // JAMAIS la 1re lecture
        assert_eq!(r.total_bytes, 200);
    }

    #[test]
    fn test_reread_after_edit_is_postedit_not_pure() {
        let mut a = AllToolsAnalyzer::new();
        a.ingest_session(
            &[
                read_ev(0, "a.rs", 100),
                edit_ev(1, "a.rs"),
                read_ev(2, "a.rs", 80),
            ],
            0,
        );
        let r = a.read_breakdown();
        assert_eq!(r.pure_reread_calls, 0);
        assert_eq!(r.postedit_reread_calls, 1);
        assert_eq!(r.postedit_reread_bytes, 80);
        assert_eq!(r.locked_bytes, 100); // la 1re lecture précède une édition
    }

    #[test]
    fn test_edit_then_first_read_is_not_a_reread() {
        let mut a = AllToolsAnalyzer::new();
        a.ingest_session(&[edit_ev(0, "a.rs"), read_ev(1, "a.rs", 50)], 0);
        let r = a.read_breakdown();
        assert_eq!(r.pure_reread_calls + r.postedit_reread_calls, 0);
    }

    #[test]
    fn test_cross_session_first_read_not_counted_as_waste() {
        // Anti-régression méthodologique (l'erreur 26% -> 6,5% de la spec §5) :
        // le même fichier lu dans 2 sessions = cross-session, PAS relecture.
        let mut a = AllToolsAnalyzer::new();
        a.ingest_session(&[read_ev(0, "a.rs", 100)], 0);
        a.ingest_session(&[read_ev(0, "a.rs", 100)], 0);
        let r = a.read_breakdown();
        assert_eq!(r.pure_reread_calls, 0);
        assert_eq!(r.cross_session_first_bytes, 100);
    }

    #[test]
    fn test_binary_extension_detected_case_insensitive() {
        let mut a = AllToolsAnalyzer::new();
        a.ingest_session(&[read_ev(0, "C:\\Docs\\Contrat.PDF", 500)], 0);
        assert_eq!(a.read_breakdown().binary_bytes, 500);
    }

    #[test]
    fn test_edit_on_other_path_does_not_interfere() {
        let mut a = AllToolsAnalyzer::new();
        a.ingest_session(
            &[
                read_ev(0, "a.rs", 10),
                edit_ev(1, "b.rs"),
                read_ev(2, "a.rs", 10),
            ],
            0,
        );
        let r = a.read_breakdown();
        assert_eq!(r.pure_reread_calls, 1);
        assert_eq!(r.locked_bytes, 0);
    }

    #[test]
    fn test_finish_sorts_families_and_limits_offenders() {
        let mut a = AllToolsAnalyzer::new();
        let bash = ToolEvent {
            kind: ToolEventKind::Result {
                tool: "Bash".into(),
                bytes: 10,
                command: Some("git status".into()),
                file_path: None,
            },
            seq: 0,
        };
        let mcp = ToolEvent {
            kind: ToolEventKind::Result {
                tool: "mcp__grafana__query_prometheus".into(),
                bytes: 999,
                command: None,
                file_path: None,
            },
            seq: 1,
        };
        a.ingest_session(
            &[bash, mcp, read_ev(2, "a.rs", 100), read_ev(3, "b.rs", 50)],
            2,
        );
        let r = a.finish(1);
        assert_eq!(r.total_bytes, 10 + 999 + 150);
        assert_eq!(r.unmatched_results, 2);
        assert_eq!(r.families[0].family, "MCP:grafana"); // plus gros d'abord
        assert!(r.families.iter().any(|f| f.family == "Bash (raw)"));
        assert_eq!(r.offenders.len(), 1); // limit=1
        assert_eq!(r.offenders[0].key, "MCP:grafana"); // 999 > 100
    }

    #[test]
    fn test_finish_bash_rtk_vs_raw_split() {
        let mut a = AllToolsAnalyzer::new();
        let mk = |seq: usize, cmd: &str| ToolEvent {
            kind: ToolEventKind::Result {
                tool: "Bash".into(),
                bytes: 1,
                command: Some(cmd.into()),
                file_path: None,
            },
            seq,
        };
        a.ingest_session(
            &[
                mk(0, "rtk git status"),
                mk(1, "cd x && rtk ls"),
                mk(2, "git log"),
            ],
            0,
        );
        let r = a.finish(10);
        let get = |name: &str| {
            r.families
                .iter()
                .find(|f| f.family == name)
                .map(|f| f.calls)
        };
        assert_eq!(get("Bash (rtk)"), Some(2));
        assert_eq!(get("Bash (raw)"), Some(1));
    }

    #[test]
    fn test_finish_task_and_grep_family_names() {
        // Attribution mandatée par la revue Task 2 : Task/Agent regroupé, Grep inchangé.
        let mut a = AllToolsAnalyzer::new();
        let task_ev = ToolEvent {
            kind: ToolEventKind::Result {
                tool: "Task".into(),
                bytes: 5,
                command: None,
                file_path: None,
            },
            seq: 0,
        };
        let grep_ev = ToolEvent {
            kind: ToolEventKind::Result {
                tool: "Grep".into(),
                bytes: 3,
                command: None,
                file_path: None,
            },
            seq: 1,
        };
        a.ingest_session(&[task_ev, grep_ev], 0);
        let r = a.finish(10);
        let get = |name: &str| {
            r.families
                .iter()
                .find(|f| f.family == name)
                .map(|f| f.calls)
        };
        assert_eq!(get("Task/Agent"), Some(1));
        assert_eq!(get("Grep"), Some(1));
    }
}
