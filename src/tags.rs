//! Tag names as a collapsible tree.
//!
//! Scalar tags are paths — `data_profiler/swh_content-dedup_bd912224/mean_loss`
//! — and a real run has dozens of them sharing long prefixes. Listed flat,
//! every row repeats the prefix and the part that tells them apart falls off
//! the right edge of the sidebar. Split on `/`, the shared prefix becomes one
//! row and each row carries only its own segment.
//!
//! Two shaping rules keep the tree shallow:
//!
//! * A chain of single-child groups collapses into one row (`a/b/c` rather
//!   than three rows of one child each) — depth costs indent, and indent is
//!   width the names need.
//! * Groups at the top start open and deeper ones closed, so the first thing
//!   on screen is the set of groups rather than every leaf at once.

use std::collections::{BTreeMap, HashMap};

/// One line of the tag list: a group to open, or a tag to plot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    pub depth: usize,
    /// What this row shows: its own segment, or several joined by `/` when a
    /// single-child chain was compressed.
    pub label: String,
    /// Full path from the root; the tag itself for a leaf, and the key the
    /// open/closed state is stored under.
    pub path: String,
    pub leaf: bool,
    /// Tags at or below this row.
    pub leaves: usize,
    pub expanded: bool,
}

#[derive(Default)]
struct Node {
    children: BTreeMap<String, Node>,
    /// Set when a tag ends exactly here.
    is_tag: bool,
    leaves: usize,
    /// Segments this node stands for, after compression.
    label: String,
}

impl Node {
    fn insert(&mut self, segments: &[&str]) {
        self.leaves += 1;
        match segments {
            [] => self.is_tag = true,
            [head, rest @ ..] => {
                let child = self.children.entry((*head).to_string()).or_default();
                if child.label.is_empty() {
                    child.label = (*head).to_string();
                }
                child.insert(rest);
            }
        }
    }

    /// Fold `a -> b -> c` into one row when each step has nothing else in it.
    /// A node that is itself a tag stops the fold: it has to stay selectable.
    fn compress(&mut self) {
        while !self.is_tag && self.children.len() == 1 {
            let (_, child) = self.children.iter().next().unwrap();
            if child.is_tag && !child.children.is_empty() {
                break; // both a tag and a group: keep it on its own row
            }
            let (key, child) = self.children.pop_first().unwrap();
            self.label =
                if self.label.is_empty() { key } else { format!("{}/{}", self.label, key) };
            self.children = child.children;
            self.is_tag = child.is_tag;
        }
        for child in self.children.values_mut() {
            child.compress();
        }
    }
}

/// Whether a group is open. `state` holds the explicit choices; anything not
/// in it follows the default of open at the top level, closed below.
pub fn is_expanded(state: &HashMap<String, bool>, path: &str, depth: usize) -> bool {
    *state.get(path).unwrap_or(&(depth == 0))
}

/// Flatten sorted tags into the rows the sidebar draws.
///
/// `force_open` ignores the collapse state — used while a filter is active, so
/// a match is never hidden inside a closed group.
pub fn rows(tags: &[String], state: &HashMap<String, bool>, force_open: bool) -> Vec<Row> {
    let mut root = Node::default();
    for tag in tags {
        let segments: Vec<&str> = tag.split('/').filter(|s| !s.is_empty()).collect();
        if segments.is_empty() {
            continue;
        }
        root.insert(&segments);
    }
    for child in root.children.values_mut() {
        child.compress();
    }
    let mut out = Vec::new();
    walk(&root, 0, "", state, force_open, &mut out);
    out
}

fn walk(
    node: &Node,
    depth: usize,
    prefix: &str,
    state: &HashMap<String, bool>,
    force_open: bool,
    out: &mut Vec<Row>,
) {
    for child in node.children.values() {
        let path = if prefix.is_empty() {
            child.label.clone()
        } else {
            format!("{}/{}", prefix, child.label)
        };
        let leaf = child.children.is_empty();
        let expanded = !leaf && (force_open || is_expanded(state, &path, depth));
        out.push(Row {
            depth,
            label: child.label.clone(),
            path: path.clone(),
            leaf,
            leaves: child.leaves,
            expanded,
        });
        if expanded {
            walk(child, depth + 1, &path, state, force_open, out);
        }
    }
}

/// Index into `tags` of the first tag at or under `path`.
///
/// Group rows plot the first tag they contain, so moving over a closed group
/// still shows something.
pub fn first_leaf(tags: &[String], row: &Row) -> Option<usize> {
    if row.leaf {
        return tags.iter().position(|t| *t == row.path);
    }
    let prefix = format!("{}/", row.path);
    tags.iter().position(|t| t.starts_with(&prefix))
}

/// Shorten a label to `width`, cutting the middle rather than the end.
///
/// Tag names differ at the tail as often as at the head — `..._bd912224` and
/// `..._524cb72` are the same until the hash — so dropping the end would make
/// separate groups look identical.
pub fn elide(label: &str, width: usize) -> String {
    let chars: Vec<char> = label.chars().collect();
    if chars.len() <= width {
        return label.to_string();
    }
    if width <= 1 {
        return "…".repeat(width);
    }
    let keep = width - 1;
    let head = keep.div_ceil(2);
    let tail = keep - head;
    let mut s: String = chars[..head].iter().collect();
    s.push('…');
    s.extend(&chars[chars.len() - tail..]);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tags(list: &[&str]) -> Vec<String> {
        let mut v: Vec<String> = list.iter().map(|s| s.to_string()).collect();
        v.sort();
        v
    }

    fn labels(rows: &[Row]) -> Vec<String> {
        rows.iter().map(|r| format!("{}{}", "  ".repeat(r.depth), r.label)).collect()
    }

    #[test]
    fn a_shared_prefix_becomes_one_row() {
        let t = tags(&["train/loss", "train/accuracy", "val/loss"]);
        let list = rows(&t, &HashMap::new(), false);
        assert_eq!(
            labels(&list),
            // `val` holds one tag, so it folds into a single row by the
            // single-child rule; `train` holds two and stays a group
            ["train", "  accuracy", "  loss", "val/loss"],
            "top level open, leaves under it"
        );
        assert_eq!(list[0].leaves, 2);
        assert!(!list[0].leaf && list[1].leaf);
    }

    #[test]
    fn deeper_groups_start_closed() {
        let t = tags(&[
            "data_profiler/legal_524cb72/mean_loss",
            "data_profiler/legal_524cb72/std_loss",
            "data_profiler/swh_bd912224/mean_loss",
            "data_profiler/swh_bd912224/std_loss",
        ]);
        let list = rows(&t, &HashMap::new(), false);
        // 40 tags of this shape used to be 40 rows; now the second level is
        // the whole list until something is opened
        assert_eq!(labels(&list), ["data_profiler", "  legal_524cb72", "  swh_bd912224"]);
        assert_eq!(list[1].leaves, 2);

        let mut state = HashMap::new();
        state.insert("data_profiler/swh_bd912224".to_string(), true);
        let list = rows(&t, &state, false);
        assert_eq!(
            labels(&list),
            ["data_profiler", "  legal_524cb72", "  swh_bd912224", "    mean_loss", "    std_loss"]
        );
    }

    #[test]
    fn closing_the_top_hides_everything_under_it() {
        let t = tags(&["train/loss", "train/acc"]);
        let mut state = HashMap::new();
        state.insert("train".to_string(), false);
        let list = rows(&t, &state, false);
        assert_eq!(labels(&list), ["train"]);
        assert!(!list[0].expanded);
    }

    #[test]
    fn a_single_child_chain_collapses_into_one_row() {
        let t = tags(&["a/b/c/metric", "a/b/c/other"]);
        let list = rows(&t, &HashMap::new(), false);
        assert_eq!(labels(&list), ["a/b/c", "  metric", "  other"], "no ladder of single children");
        assert_eq!(list[0].path, "a/b/c");
    }

    #[test]
    fn a_lone_tag_is_one_row_not_a_group() {
        let t = tags(&["some/deep/lonely_metric"]);
        let list = rows(&t, &HashMap::new(), false);
        assert_eq!(labels(&list), ["some/deep/lonely_metric"]);
        assert!(list[0].leaf);
        assert_eq!(list[0].path, "some/deep/lonely_metric");
    }

    #[test]
    fn a_tag_that_is_also_a_group_keeps_its_own_row() {
        let t = tags(&["loss", "loss/train", "loss/val"]);
        let list = rows(&t, &HashMap::new(), false);
        assert_eq!(labels(&list), ["loss", "  train", "  val"]);
        assert!(!list[0].leaf, "it has children, so it stays a group row");
    }

    #[test]
    fn tags_without_slashes_are_plain_rows() {
        let t = tags(&["loss", "accuracy"]);
        let list = rows(&t, &HashMap::new(), false);
        assert_eq!(labels(&list), ["accuracy", "loss"]);
        assert!(list.iter().all(|r| r.leaf));
    }

    #[test]
    fn a_filter_opens_everything() {
        let t = tags(&["a/b/x_loss", "a/c/y_loss"]);
        let closed: HashMap<String, bool> = [("a".to_string(), false)].into_iter().collect();
        let list = rows(&t, &closed, true);
        assert_eq!(labels(&list), ["a", "  b/x_loss", "  c/y_loss"], "a match is never hidden");
    }

    #[test]
    fn a_group_row_points_at_its_first_tag() {
        let t = tags(&["p/g1/a_loss", "p/g1/b_loss", "p/g2/c_loss"]);
        let list = rows(&t, &HashMap::new(), false);
        let group = list.iter().find(|r| r.path == "p/g1").unwrap();
        assert_eq!(first_leaf(&t, group), Some(0));
        assert_eq!(t[0], "p/g1/a_loss");
        let leafrow = Row { leaf: true, path: "p/g2/c_loss".into(), ..group.clone() };
        assert_eq!(first_leaf(&t, &leafrow), Some(2));
    }

    #[test]
    fn eliding_keeps_both_ends() {
        // the hash at the end is what tells two groups apart
        let out = elide("swh_content-dedup-opc-filtered_bd912224", 20);
        assert_eq!(out, "swh_conten…_bd912224");
        assert_eq!(out.chars().count(), 20);
        assert!(out.ends_with("_bd912224"), "the hash that names the group survives");
        assert_eq!(elide("short", 20), "short");
        assert_eq!(elide("exactly_ten", 11), "exactly_ten");
        assert_eq!(elide("abcdef", 3), "a…f");
    }

    #[test]
    fn eliding_is_never_wider_than_asked() {
        for w in 0..12 {
            let out = elide("a_rather_long_label_here", w);
            assert!(out.chars().count() <= w.max(0), "width {} gave {:?}", w, out);
        }
    }
}
