from codebase import causalityTable
from pathlib import Path
from anytree import Node, RenderTree
import polars as pl
import numpy as np


class KBVisualisation:
    def __init__(self, kb_path: Path, tree_path: Path):
        self.kb = causalityTable.CausalityTable(kb_path)
        self.tree = causalityTable.CausalityTable(tree_path)

        self.kb.load_data(date_parsing=False)
        self.tree.load_data(date_parsing=False)

        self.markers_tree_df =  self.kb.df.join(self.tree.df, left_on="marker", right_on="marker", how="left").select(
            "id","marker","label","markerType","parentMarker","displayMarker","children")
        
    def visualize(self, marker:str, depth=10):
        racine_df = self.markers_tree_df.filter(pl.col("marker") == marker)
        print("Visualizing tree for marker:", marker)
        if racine_df.is_empty():
            raise ValueError(f"Marker {marker} not found in the KB.")
        else:
            racine = racine_df[0]
            print("Parent Marker:", racine["parentMarker"][0])
            root = Node(f"{racine['marker'][0]} ({racine['label'][0]})")
            children = np.setdiff1d(racine['children'][0], racine['marker'][0])
            self._add_children(root, children, depth)
            for pre, _, node in RenderTree(root):
                print(f"{pre}{node.name}")

    def _add_children(self, parent_node, children_markers, depth):
        if children_markers is None or len(children_markers) == 0 or depth == 0:
            return
        for child_marker in children_markers:
            child_df = self.markers_tree_df.filter(pl.col("marker") == child_marker)
            if not child_df.is_empty():
                child = child_df[0]
                child_node = Node(f"{child['marker'][0]} ({child['label'][0]})", parent=parent_node)
                children = np.setdiff1d(child['children'][0], child['marker'][0])
                self._add_children(child_node, children, depth - 1)

            
        return
            

if __name__ == "__main__":
    kb_path = Path("data/causalitylink_sample/KB")
    tree_path = Path("data/causalitylink_sample/Tree")
    visualisation = KBVisualisation(kb_path, tree_path)

    marker = ""
    prof = 2
    while marker != "__exit__":
        print("\n\n\n------------------------------------------------------------------------------ \n\n")
        input_marker = input("Enter marker to visualize (or '__exit__' to quit - type #N to set depth to N): ")
        if '#' in input_marker:
            marker = input_marker.split('#')[0].strip()
            prof = int(input_marker.split('#')[1])
            print(f"Setting depth to {prof}")
        else:
            marker = input_marker.strip()
        print("\n\n\n------------------------------------------------------------------------------ \n\n")
        try:
            visualisation.visualize(marker, depth=prof)
        except ValueError as e:
            print("Error:", e)

