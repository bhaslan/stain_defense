using PhyloTree;
using System.Collections.Generic;
using TreeViewer;
using System;

namespace aCladeColours
{
	public static class CustomCode
	{
		const string cladeAttribute = "clade";
		const string colourAttribute = "clade_colour";
		const string backboneColour = "#9E9E9E";
		const string labelAttribute = "support_label";
		const int minLabelLeaves = 20;
		static readonly string[] palette = { PALETTE };

		public static void PerformAction(ref TreeNode tree, TreeCollection trees, InstanceStateData stateData, Action<double> progressAction)
		{
			List<TreeNode> nodes = new List<TreeNode>(tree.GetChildrenRecursive());
			if (!nodes.Contains(tree)) nodes.Add(tree);

			Dictionary<TreeNode, int> codes = new Dictionary<TreeNode, int>();
			foreach (TreeNode node in nodes) codes[node] = Code(node);

			foreach (TreeNode node in nodes)
			{
				int code = codes[node];
				node.Attributes[colourAttribute] = code >= 0 && code < palette.Length ? palette[code] : backboneColour;
				if (Major(node, tree, codes)) node.Attributes[labelAttribute] = Support(node);
			}
		}

		static bool Major(TreeNode node, TreeNode root, Dictionary<TreeNode, int> codes)
		{
			if (node == root || node.Children.Count == 0 || node.GetLeaves().Count < minLabelLeaves) return false;
			if (codes[node] >= 0) return codes[node.Parent] == -1;
			if (node.Parent == root && root.Children.Count == 2)
			{
				TreeNode sibling = root.Children[0] == node ? root.Children[1] : root.Children[0];
				if (codes[sibling] >= 0 || root.Children[0] != node) return false;
			}
			return true;
		}

		static string Support(TreeNode node)
		{
			return node.Attributes.TryGetValue("Support2", out object value) && value is double ufboot ? Math.Round(ufboot).ToString("0") : "?";
		}

		static int Code(TreeNode node)
		{
			int code = -1;

			foreach (TreeNode leaf in node.GetLeaves())
			{
				if (!leaf.Attributes.TryGetValue(cladeAttribute, out object value) || !(value is double number)) return -1;
				if (code == -1) code = (int)number;
				else if (code != (int)number) return -1;
			}

			return code;
		}
	}
}
