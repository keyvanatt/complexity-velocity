"""Interactive explorer for the CausalityLink knowledge-base marker hierarchy.

Usage:
    python kb_visualisation.py [--data-dir data/causalitylink_sample]

At the prompt, type a marker name to render its subtree. Append ``#N`` to set
the rendering depth (e.g. ``sport#3``). Type ``__exit__`` to quit.
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from anytree import Node, RenderTree

from causalityTable import CausalityTable

DEFAULT_DATA_DIR = Path("data/causalitylink_sample")


class KBVisualisation:
    def __init__(self, kb_path: Path, tree_path: Path):
        self.kb = CausalityTable(kb_path)
        self.tree = CausalityTable(tree_path)

        self.kb.load_data(date_parsing=False)
        self.tree.load_data(date_parsing=False)

        self.markers_tree_df = self.kb.df.join(
            self.tree.df, left_on="marker", right_on="marker", how="left"
        ).select(
            "id", "marker", "label", "markerType", "parentMarker", "displayMarker", "children"
        )

    def visualize(self, marker: str, depth: int = 10):
        racine_df = self.markers_tree_df.filter(pl.col("marker") == marker)
        print("Visualizing tree for marker:", marker)
        if racine_df.is_empty():
            raise ValueError(f"Marker {marker} not found in the KB.")

        racine = racine_df[0]
        print("Parent Marker:", racine["parentMarker"][0])
        root = Node(f"{racine['marker'][0]} ({racine['label'][0]})")
        children = np.setdiff1d(racine["children"][0], racine["marker"][0])
        self._add_children(root, children, depth)
        for pre, _, node in RenderTree(root):
            print(f"{pre}{node.name}")

    def _add_children(self, parent_node, children_markers, depth):
        if children_markers is None or len(children_markers) == 0 or depth == 0:
            return
        for child_marker in children_markers:
            child_df = self.markers_tree_df.filter(pl.col("marker") == child_marker)
            if child_df.is_empty():
                continue
            child = child_df[0]
            child_node = Node(f"{child['marker'][0]} ({child['label'][0]})", parent=parent_node)
            children = np.setdiff1d(child["children"][0], child["marker"][0])
            self._add_children(child_node, children, depth - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory holding the KB/ and Tree/ AVRO folders (default: %(default)s)",
    )
    parser.add_argument("--depth", type=int, default=2, help="Initial rendering depth")
    args = parser.parse_args()

    visualisation = KBVisualisation(args.data_dir / "KB", args.data_dir / "Tree")

    separator = "\n\n\n" + "-" * 78 + "\n\n"
    marker = ""
    depth = args.depth
    while marker != "__exit__":
        print(separator)
        user_input = input(
            "Enter marker to visualize (or '__exit__' to quit - type #N to set depth to N): "
        )
        if "#" in user_input:
            marker, _, depth_str = user_input.partition("#")
            marker = marker.strip()
            depth = int(depth_str)
            print(f"Setting depth to {depth}")
        else:
            marker = user_input.strip()
        if marker == "__exit__":
            break
        print(separator)
        try:
            visualisation.visualize(marker, depth=depth)
        except ValueError as e:
            print("Error:", e)


if __name__ == "__main__":
    main()
