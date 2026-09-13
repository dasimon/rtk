//! All-tools token attribution: extracts every tool_use/tool_result pair
//! from Claude Code session JSONL (the Bash-only provider misses ~65% of bytes).

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug)]
pub enum ToolEventKind {
    /// A tool_result matched with its tool_use.
    Result {
        tool: String,
        bytes: usize,
        /// Bash only: the command as emitted (before any hook rewrite).
        command: Option<String>,
        /// Normalized path, if the tool provides one (Read in particular).
        file_path: Option<String>,
    },
    /// An Edit/Write/NotebookEdit tool_use (counted when emitted, not at its result).
    FileEdit { file_path: String },
}

#[derive(Debug)]
pub struct ToolEvent {
    pub kind: ToolEventKind,
    pub seq: usize,
}

/// Lowercase + unified separators: Windows transcripts mix both.
pub fn normalize_path(p: &str) -> String {
    p.replace('\\', "/").to_lowercase()
}

const EDIT_TOOLS: &[&str] = &["Edit", "Write", "NotebookEdit"];

/// Extracts every tool event of a session, in file order.
/// Returns (events, number of truly orphaned tool_results — a tool_result whose
/// tool_use is an Edit/Write/NotebookEdit is consumed silently, not counted,
/// since its event was already emitted at the tool_use). Invalid lines are
/// skipped.
pub fn extract_tool_events(path: &Path) -> Result<(Vec<ToolEvent>, usize)> {
    let file =
        std::fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = BufReader::new(file);

    // id -> (tool name, optional Bash command, optional Read file_path)
    let mut pending: HashMap<String, (String, Option<String>, Option<String>)> = HashMap::new();
    // Edit/Write/NotebookEdit tool_use ids -> tool name: their tool_result is
    // expected and must now be emitted as a Result event (the result's bytes),
    // in addition to the FileEdit already emitted at tool_use (spec §4.1: Edit/
    // Write/NotebookEdit must show up as families in the ALL TOOLS table).
    let mut edit_ids: HashMap<String, String> = HashMap::new();
    let mut events = Vec::new();
    let mut unmatched = 0usize;
    let mut seq = 0usize;

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        // Pre-filter: "tool_use\"" matches the use blocks' type but not "tool_use_id".
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
                            edit_ids.insert(id.to_string(), name.to_string());
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
                    if let Some(tool) = edit_ids.remove(id) {
                        // Result of an Edit/Write/NotebookEdit: the FileEdit was
                        // already emitted at tool_use, but the result's bytes are
                        // still counted here (spec §4.1) so the family shows up in
                        // the ALL TOOLS table and total_bytes is not understated.
                        events.push(ToolEvent {
                            kind: ToolEventKind::Result {
                                tool,
                                bytes: result_bytes(block),
                                command: None,
                                file_path: None,
                            },
                            seq,
                        });
                        seq += 1;
                        continue;
                    }
                    let Some((tool, command, file_path)) = pending.remove(id) else {
                        unmatched += 1;
                        continue;
                    };
                    let bytes = result_bytes(block);
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

/// Size of a tool_result: length of `content` if it is a string, otherwise the
/// length of its serialized JSON (spec §5).
fn result_bytes(block: &serde_json::Value) -> usize {
    match block.get("content") {
        Some(serde_json::Value::String(s)) => s.len(),
        Some(other) => serde_json::to_string(other).map(|s| s.len()).unwrap_or(0),
        None => 0,
    }
}

/// Extensions whose read content is binary/extracted: no filter can compress it.
const BINARY_EXTS: &[&str] = &[
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".xls", ".xlsx", ".doc", ".docx",
    ".ppt", ".pptx", ".zip",
];

/// Expects a path already normalized by `normalize_path` (lowercase, `/`).
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
struct OffenderStat {
    bytes: usize,
    calls: usize,
    max_calls_one_session: usize,
    edited: bool,
    binary: bool,
}

pub struct AllToolsAnalyzer {
    families: HashMap<String, (usize, usize)>, // family -> (calls, bytes)
    read: ReadBreakdown,
    offenders: HashMap<String, OffenderStat>,
    /// How many EARLIER sessions read this path.
    seen_paths: HashMap<String, usize>,
    unmatched_results: usize,
}

impl Default for AllToolsAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl AllToolsAnalyzer {
    pub fn new() -> Self {
        Self {
            families: HashMap::new(),
            read: ReadBreakdown::default(),
            offenders: HashMap::new(),
            seen_paths: HashMap::new(),
            unmatched_results: 0,
        }
    }

    #[doc(hidden)]
    #[allow(dead_code)] // exposed for tests only (read_breakdown); finish() reads self.read directly
    pub fn read_breakdown(&self) -> &ReadBreakdown {
        &self.read
    }

    fn family_key(&self, tool: &str, command: Option<&str>) -> String {
        if tool == "Bash" {
            // Reuse the same chain-splitter mod.rs's Bash pass already relies on
            // (registry::split_command_chain) instead of a separate regex, so this
            // classification can never contradict `already_rtk` on the same output
            // and never false-positives on "rtk" inside a quoted string.
            let rtk = command
                .map(|c| {
                    crate::discover::registry::split_command_chain(c)
                        .iter()
                        .any(|p| p.trim_start().starts_with("rtk ") || p.trim() == "rtk")
                })
                .unwrap_or(false);
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
        // Pass 1: edit positions per path.
        let mut edit_seqs: HashMap<&str, Vec<usize>> = HashMap::new();
        for e in events {
            if let ToolEventKind::FileEdit { file_path } = &e.kind {
                edit_seqs.entry(file_path.as_str()).or_default().push(e.seq);
            }
        }
        // Pass 2: results.
        let mut last_read_seq: HashMap<&str, usize> = HashMap::new();
        // Read paths only: feeds seen_paths (cross-session).
        let mut session_reads: HashMap<&str, usize> = HashMap::new();
        // ALL offender keys (Read paths + MCP:<server>): feeds
        // max_calls_one_session for every offender, not just Read.
        let mut session_offender_calls: HashMap<String, usize> = HashMap::new();
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
                *session_offender_calls.entry(key.clone()).or_insert(0) += 1;
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
        // End of session: cross-session (Read paths only).
        for p in session_reads.keys() {
            *self.seen_paths.entry(p.to_string()).or_insert(0) += 1;
        }
        // End of session: per-session max, for ALL offenders (Read + MCP).
        for (key, n) in session_offender_calls {
            if let Some(o) = self.offenders.get_mut(&key) {
                o.max_calls_one_session = o.max_calls_one_session.max(n);
            }
        }
    }

    /// Consumes the analyzer and produces the aggregated report: families sorted
    /// by descending bytes, top `limit` offenders (same order), totals.
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
pub struct FamilyStat {
    pub family: String,
    pub calls: usize,
    pub bytes: usize,
}

#[derive(Debug, serde::Serialize)]
pub struct OffenderEntry {
    pub key: String,
    pub bytes: usize,
    pub calls: usize,
    pub max_calls_one_session: usize,
    pub edited: bool,
    pub binary: bool,
}

#[derive(Debug, serde::Serialize)]
pub struct AllToolsReport {
    pub families: Vec<FamilyStat>,
    /// Read details (see `ReadBreakdown`).
    /// ⚠️ Known reconciliation gap: a Read tool_result without `file_path`
    /// (e.g. a Read denied or failing before the path is resolved) is counted in
    /// `families["Read"]` (calls/bytes) but NOT in this breakdown, which needs
    /// a path to classify the read (reread, locked, binary, etc.).
    /// `read.total_bytes` can therefore be strictly lower than the bytes of the
    /// "Read" family.
    pub read: ReadBreakdown,
    pub offenders: Vec<OffenderEntry>,
    pub unmatched_results: usize,
    pub total_bytes: usize,
}

fn fmt_bytes(b: usize) -> String {
    if b >= 1_048_576 {
        format!("{:.1} MB", b as f64 / 1_048_576.0)
    } else if b >= 1024 {
        format!("{:.1} KB", b as f64 / 1024.0)
    } else {
        format!("{} B", b)
    }
}

fn pct(part: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        part as f64 * 100.0 / total as f64
    }
}

/// Text rendering of the `--all-tools` report (full attribution, all tools).
pub fn format_all_tools_text(r: &AllToolsReport) -> String {
    let mut out = String::with_capacity(2048);
    out.push_str("ALL TOOLS -- Token Attribution (tool_result bytes, ~4 bytes/token)\n");
    out.push_str(&"=".repeat(68));
    out.push('\n');
    out.push_str(&format!(
        "{:<28}{:>8}{:>12}{:>8}\n",
        "FAMILY", "CALLS", "BYTES", "SHARE"
    ));
    for f in &r.families {
        out.push_str(&format!(
            "{:<28}{:>8}{:>12}{:>7.1}%\n",
            f.family,
            f.calls,
            fmt_bytes(f.bytes),
            pct(f.bytes, r.total_bytes)
        ));
    }
    let rd = &r.read;
    out.push_str("\nREAD BREAKDOWN (percentages of Read bytes; classes overlap)\n");
    out.push_str(&format!(
        "  binary (pdf/img/office)        {:>6.1}%  incompressible\n",
        pct(rd.binary_bytes, rd.total_bytes)
    ));
    out.push_str(&format!(
        "  re-read same session, no edit  {:>6.1}%  deduplicable ({} calls)\n",
        pct(rd.pure_reread_bytes, rd.total_bytes),
        rd.pure_reread_calls
    ));
    out.push_str(&format!(
        "  re-read after edit (verify)    {:>6.1}%  ({} calls)\n",
        pct(rd.postedit_reread_bytes, rd.total_bytes),
        rd.postedit_reread_calls
    ));
    out.push_str(&format!(
        "  read then edited (locked)      {:>6.1}%  never filterable\n",
        pct(rd.locked_bytes, rd.total_bytes)
    ));
    out.push_str(&format!(
        "  first read in later session    {:>6.1}%  legitimate\n",
        pct(rd.cross_session_first_bytes, rd.total_bytes)
    ));
    out.push_str("\nTOP OFFENDERS\n");
    for o in &r.offenders {
        let mut tags: Vec<String> = Vec::new();
        if o.edited {
            tags.push("[EDITED]".to_string());
        }
        if o.binary {
            tags.push("[BINARY]".to_string());
        }
        if o.calls > 1 {
            tags.push(format!(
                "[x{} max {}/session]",
                o.calls, o.max_calls_one_session
            ));
        }
        out.push_str(&format!(
            "  {:>10}  {} {}\n",
            fmt_bytes(o.bytes),
            o.key,
            tags.join(" ")
        ));
    }
    let rtk_bytes = r
        .families
        .iter()
        .filter(|f| f.family == "Bash (rtk)")
        .map(|f| f.bytes)
        .sum::<usize>();
    out.push_str(&format!(
        "\nVERDICT: {:.0}% incompressible (binary) | {:.1}% deduplicable (pure re-reads) | \
         {:.0}% locked by edits | {:.0}% already through rtk\n",
        pct(rd.binary_bytes, r.total_bytes),
        pct(rd.pure_reread_bytes, r.total_bytes),
        pct(rd.locked_bytes, r.total_bytes),
        pct(rtk_bytes, r.total_bytes)
    ));
    if r.unmatched_results > 0 {
        out.push_str(&format!(
            "({} unmatched tool results ignored)\n",
            r.unmatched_results
        ));
    }
    out
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
                assert_eq!(file_path.as_deref(), Some("c:/proj/a.rs")); // normalized
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
                // length of serde_json::to_string of the content array
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
        // Finding A: an Edit's tool_result is still emitted as a Result event
        // (its bytes must count toward total_bytes / the ALL TOOLS table), in
        // addition to the FileEdit already emitted at tool_use.
        let edit_l = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"e1","name":"Edit","input":{"file_path":"C:\\Proj\\A.rs","old_string":"x","new_string":"y"}}]}}"#;
        let edit_res = r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"e1","content":"ignored"}]}}"#;
        let f = make_jsonl(&[edit_l, edit_res]);
        let (events, unmatched) = extract_tool_events(f.path()).unwrap();
        assert_eq!(unmatched, 0);
        assert_eq!(events.len(), 2);
        match &events[0].kind {
            ToolEventKind::FileEdit { file_path } => assert_eq!(file_path, "c:/proj/a.rs"),
            _ => panic!("expected FileEdit"),
        }
        match &events[1].kind {
            ToolEventKind::Result {
                tool,
                bytes,
                command,
                file_path,
            } => {
                assert_eq!(tool, "Edit");
                assert_eq!(*bytes, "ignored".len());
                assert!(command.is_none());
                assert!(file_path.is_none());
            }
            _ => panic!("expected Result"),
        }
    }

    #[test]
    fn test_finish_edit_family_carries_result_bytes() {
        // Finding A, finish()-level: the "Edit" family appears with the result's
        // bytes (denominator of SHARE/VERDICT percentages must not be understated).
        let edit_l = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"e1","name":"Edit","input":{"file_path":"C:\\Proj\\A.rs","old_string":"x","new_string":"y"}}]}}"#;
        let edit_res = r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"e1","content":"ignored"}]}}"#;
        let f = make_jsonl(&[edit_l, edit_res]);
        let (events, unmatched) = extract_tool_events(f.path()).unwrap();
        let mut a = AllToolsAnalyzer::new();
        a.ingest_session(&events, unmatched);
        let r = a.finish(10);
        let edit_family = r.families.iter().find(|f| f.family == "Edit");
        assert!(edit_family.is_some(), "expected an Edit family entry");
        let edit_family = edit_family.unwrap();
        assert_eq!(edit_family.calls, 1);
        assert_eq!(edit_family.bytes, "ignored".len());
        assert_eq!(r.total_bytes, "ignored".len());
    }

    #[test]
    fn test_extract_bash_carries_command_and_seq_order() {
        let bash_use = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t4","name":"Bash","input":{"command":"rtk git status"}}]}}"#;
        let bash_res = r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t4","content":"ok"}]}}"#;
        let edit_l = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t5","name":"Write","input":{"file_path":"b.txt","content":"z"}}]}}"#;
        let f = make_jsonl(&[bash_use, bash_res, edit_l]);
        let (events, _) = extract_tool_events(f.path()).unwrap();
        assert_eq!(events.len(), 2);
        // the Bash result comes before the Write use in the file -> increasing seq
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
        assert_eq!(r.pure_reread_bytes, 100); // NEVER the first read
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
        assert_eq!(r.locked_bytes, 100); // the first read precedes an edit
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
        // Methodological regression guard (the 26% -> 6.5% error from spec §5):
        // the same file read in 2 sessions = cross-session, NOT a re-read.
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
        assert_eq!(r.families[0].family, "MCP:grafana"); // largest first
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
    fn test_finish_bash_rtk_quoted_string_false_positive_rejected() {
        // Finding C: "rtk" appearing inside a quoted string argument must NOT be
        // classified as "Bash (rtk)" -- split_command_chain only splits on
        // unquoted operators, it doesn't scan inside string literals.
        let mut a = AllToolsAnalyzer::new();
        let ev = ToolEvent {
            kind: ToolEventKind::Result {
                tool: "Bash".into(),
                bytes: 1,
                command: Some(r#"git commit -m "use rtk always""#.into()),
                file_path: None,
            },
            seq: 0,
        };
        a.ingest_session(&[ev], 0);
        let r = a.finish(10);
        let get = |name: &str| {
            r.families
                .iter()
                .find(|f| f.family == name)
                .map(|f| f.calls)
        };
        assert_eq!(get("Bash (raw)"), Some(1));
        assert_eq!(get("Bash (rtk)"), None);
    }

    #[test]
    fn test_finish_max_calls_one_session_tracked_for_mcp_offenders() {
        // Finding B: session_reads previously only fed Read offenders (the
        // `if tool != "Read" { continue; }` guard skipped MCP entirely), so
        // MCP offenders always showed "max 0/session". Two MCP calls in the
        // SAME session must now yield max_calls_one_session == 2.
        let mut a = AllToolsAnalyzer::new();
        let mk = |seq: usize| ToolEvent {
            kind: ToolEventKind::Result {
                tool: "mcp__ms365__chat_message_search".into(),
                bytes: 10,
                command: None,
                file_path: None,
            },
            seq,
        };
        a.ingest_session(&[mk(0), mk(1)], 0);
        let r = a.finish(10);
        let offender = r
            .offenders
            .iter()
            .find(|o| o.key == "MCP:ms365")
            .expect("expected MCP:ms365 offender");
        assert_eq!(offender.calls, 2);
        assert_eq!(offender.max_calls_one_session, 2);
    }

    #[test]
    fn test_pure_reread_across_mixed_path_separators() {
        // Finding H(b): the same file read once with backslashes and once with
        // forward slashes in the same session must classify as a re-read --
        // pins normalize_path's role end-to-end (extraction, not just the
        // standalone normalize_path unit test).
        let mut a = AllToolsAnalyzer::new();
        a.ingest_session(
            &[
                read_ev(0, "C:\\proj\\a.rs", 100),
                read_ev(1, "c:/proj/a.rs", 100),
            ],
            0,
        );
        let r = a.read_breakdown();
        assert_eq!(r.pure_reread_calls, 1);
        assert_eq!(r.pure_reread_bytes, 100);
    }

    #[test]
    fn test_format_text_sections_and_verdict() {
        let mut a = AllToolsAnalyzer::new();
        a.ingest_session(
            &[
                read_ev(0, "big.pdf", 640),
                read_ev(1, "a.rs", 100),
                read_ev(2, "a.rs", 60),
            ],
            0,
        );
        let out = format_all_tools_text(&a.finish(10));
        assert!(out.contains("ALL TOOLS"));
        assert!(out.contains("Read"));
        assert!(out.contains("binary")); // Read breakdown
        assert!(out.contains("re-read"));
        assert!(out.contains("VERDICT:"));
        assert!(out.contains("deduplicable")); // verdict wording
        assert!(out.contains("[BINARY]")); // offender tag
        assert!(!out.contains("saveable")); // no quantified savings promise
    }

    #[test]
    fn test_format_text_empty_report_no_panic() {
        let out = format_all_tools_text(&AllToolsAnalyzer::new().finish(10));
        assert!(out.contains("ALL TOOLS")); // no division by zero
    }

    #[test]
    fn test_finish_task_and_grep_family_names() {
        // Attribution required by the Task 2 review: Task/Agent grouped, Grep unchanged.
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
