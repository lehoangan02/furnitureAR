"""
Scene Graph Editor  —  furnitureAR
===================================
Interactive physics-based scene-graph editor for 3D-FRONT / EchoScene data.

Controls
--------
  Drag node         : left-click + drag
  Double-click node : rename
  Right-click       : context menu
  ＋ Node button    : add a new object node
  ⟶ Edge button    : enter edge mode → click source, then click target
  Esc               : cancel edge mode

Run
---
  python scene_graph_editor.py
"""

import argparse, json, math, os, random, threading, time
import tkinter as tk
from tkinter import simpledialog, messagebox, ttk, filedialog

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── palette ───────────────────────────────────────────────────────────────────
COLOUR_MAP = {
    "dining_chair": "#e74c3c", "chair":       "#e74c3c",
    "dining_table": "#f97316", "table":       "#f97316", "desk": "#f97316",
    "pendant_lamp": "#eab308", "lamp":        "#eab308",
    "floor":        "#94a3b8",
    "bed":          "#4a9eff",
    "sofa":         "#14b8a6",
    "wardrobe":     "#22c55e",
    "nightstand":   "#a855f7",
    "cabinet":      "#7c3aed",
    "bookshelf":    "#16a34a",
    "tv":           "#3b82f6", "monitor": "#3b82f6",
    "stool":        "#dc2626",
}
DEFAULT_COL     = "#06b6d4"
EDGE_COL        = "#475569"
EDGE_LABEL_COL  = "#f87171"
SEL_EDGE_COL    = "#fb923c"

SPATIAL_RELS = [
    "above", "below", "left", "right", "front", "behind",
    "close by", "standing on", "opposite to",
]
ALL_RELS = SPATIAL_RELS + [
    "same super category as", "smaller than", "bigger than",
    "lower than", "higher than", "same material as", "same style as",
]

# ── physics ───────────────────────────────────────────────────────────────────
SPRING_LEN   = 180      # edge spring rest length
SPRING_K     = 0.04     # spring stiffness
PAIR_REST    = 150      # node–node rest distance: closer → repel, farther → attract
PAIR_REPULSE = 30000    # steep inverse-square push when closer than PAIR_REST
PAIR_K       = 0.005    # gentle pull stiffness when farther than PAIR_REST
PAIR_MAX     = 1.5      # cap on the attraction magnitude
GRAVITY_K    = 0.0025   # inward pull toward centre (keeps system centred)
CORE_REPULSE = 14000    # outward push when inside CORE_DEAD (don't pile on core)
CORE_DEAD    = 80       # radius px: inside → push out, outside → pull in
DAMPING      = 0.82
TIMESTEP     = 0.55
NODE_R       = 40
TICK_MS      = 28

# ── theme ─────────────────────────────────────────────────────────────────────
BG      = "#0f0f17"
PANEL   = "#16161f"
CARD    = "#1e1e2e"
BORDER  = "#2a2a3e"
TEXT    = "#e2e8f0"
SUBTEXT = "#64748b"
ACCENT  = "#6366f1"


# ── colour helpers ────────────────────────────────────────────────────────────
def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _lighten(c, f=0.45):
    r, g, b = _rgb(c)
    return f"#{int(r+(255-r)*f):02x}{int(g+(255-g)*f):02x}{int(b+(255-b)*f):02x}"

def _darken(c, f=0.25):
    r, g, b = _rgb(c)
    return f"#{int(r*(1-f)):02x}{int(g*(1-f)):02x}{int(b*(1-f)):02x}"

def node_col(label: str) -> str:
    lo = label.lower()
    for k, v in COLOUR_MAP.items():
        if k in lo:
            return v
    return DEFAULT_COL


# ── data model ────────────────────────────────────────────────────────────────
class Node:
    _ctr = 0

    def __init__(self, label, nid=None, x=0.0, y=0.0):
        if nid is None:
            Node._ctr += 1
            nid = Node._ctr
        self.id     = nid
        self.label  = label
        self.colour = node_col(label)
        self.x, self.y   = float(x), float(y)
        self.vx, self.vy = 0.0, 0.0
        self.pinned = False

    @property
    def short(self):
        return f"{self.label} ({self.id})"


class Edge:
    def __init__(self, src: Node, dst: Node, label: str = ""):
        self.src, self.dst, self.label = src, dst, label


class SceneGraph:
    def __init__(self):
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self._map:  dict[int, Node] = {}

    def add_node(self, label, nid=None, x=0.0, y=0.0) -> Node:
        n = Node(label, nid, x, y)
        while n.id in self._map:
            n.id += 1000
        self.nodes.append(n)
        self._map[n.id] = n
        if n.id > Node._ctr:
            Node._ctr = n.id
        return n

    def remove_node(self, node: Node):
        self.nodes = [n for n in self.nodes if n is not node]
        self.edges = [e for e in self.edges
                      if e.src is not node and e.dst is not node]
        self._map.pop(node.id, None)

    def add_edge(self, src: Node, dst: Node, label: str = "") -> Edge:
        e = Edge(src, dst, label)
        self.edges.append(e)
        return e

    def remove_edge(self, edge: Edge):
        self.edges = [e for e in self.edges if e is not edge]

    def node_by_id(self, nid: int) -> "Node | None":
        return self._map.get(nid)

    def to_dict(self) -> dict:
        return {
            "objects": {str(n.id): n.label for n in self.nodes},
            "relationships": [
                [e.src.id, e.dst.id, i, e.label]
                for i, e in enumerate(self.edges)
            ],
        }


# ── loader ────────────────────────────────────────────────────────────────────
def load_scene(rel_json: str, scene_id: str,
               spatial_only: bool = True) -> "SceneGraph | None":
    if not os.path.exists(rel_json):
        return None
    with open(rel_json) as f:
        data = json.load(f)
    scan = next((s for s in data.get("scans", [])
                 if s["scan"] == scene_id), None)
    if not scan:
        return None

    g = SceneGraph()
    for k, v in scan["objects"].items():
        g.add_node(v, int(k))           # positions set later in _initial_placement

    for rel in scan.get("relationships", []):
        sid, did, _, rtype = rel
        if spatial_only and rtype not in SPATIAL_RELS:
            continue
        src = g.node_by_id(sid)
        dst = g.node_by_id(did)
        if src and dst:
            if rtype == "standing on" and dst.label == "floor":
                continue
            g.add_edge(src, dst, rtype)
    return g


def demo_graph() -> SceneGraph:
    g = SceneGraph()
    t  = g.add_node("dining_table", 1)
    c1 = g.add_node("dining_chair", 2)
    c2 = g.add_node("dining_chair", 3)
    c3 = g.add_node("dining_chair", 4)
    c4 = g.add_node("dining_chair", 5)
    lp = g.add_node("pendant_lamp", 6)
    fl = g.add_node("floor",        7)
    g.add_edge(lp, t,  "above");  g.add_edge(lp, c1, "above")
    g.add_edge(c1, t,  "left");   g.add_edge(c2, t,  "right")
    g.add_edge(c3, c4, "behind"); g.add_edge(c2, c4, "close by")
    g.add_edge(t,  fl, "standing on")
    return g


# ── PNG export ────────────────────────────────────────────────────────────────
def export_png(graph: SceneGraph, out_path: str, title: str = "Scene Graph"):
    if not HAS_MPL:
        messagebox.showerror("Missing", "pip install matplotlib networkx")
        return
    G = nx.DiGraph()
    for n in graph.nodes:
        G.add_node(n.id)
    for e in graph.edges:
        if G.has_edge(e.src.id, e.dst.id):
            G[e.src.id][e.dst.id]["label"] += f", {e.label}"
        else:
            G.add_edge(e.src.id, e.dst.id, label=e.label)

    labels  = {n.id: f"{n.label} ({n.id})" for n in graph.nodes}
    colours = [node_col(n.label) for n in graph.nodes]
    pos     = {n.id: (n.x, -n.y) for n in graph.nodes}

    fig, ax = plt.subplots(figsize=(12, 9))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=3200,
        node_color=colours, edgecolors="black", linewidths=1.8)
    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels,
        font_size=9, font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.6,
        arrowstyle="-|>", arrowsize=22, edge_color=EDGE_COL,
        connectionstyle="arc3,rad=0.06")
    nx.draw_networkx_edge_labels(G, pos, ax=ax,
        edge_labels=nx.get_edge_attributes(G, "label"),
        font_size=7.5, font_color=EDGE_LABEL_COL, font_weight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[export] → {out_path}")


# ── App ───────────────────────────────────────────────────────────────────────
class App:
    def __init__(self, root: tk.Tk, graph: SceneGraph,
                 scene_id: str, out_png: str):
        self.root      = root
        self.graph     = graph
        self.scene_id  = scene_id
        self.out_png   = out_png
        self.running   = True

        self.sel_node:  Node | None = None
        self.sel_edge:  Edge | None = None
        self.drag_node: Node | None = None
        self.drag_ox = self.drag_oy = 0.0

        self.edge_mode = False
        self.edge_src:  Node | None = None
        self._mx = self._my = 0.0

        # will be updated by _canvas_size() after layout
        self._cw = 1000
        self._ch = 720

        self._build_ui()
        # Wait for Tk to finish ALL layout passes, then centre nodes
        self.root.update_idletasks()
        self.root.after(80, self._initial_placement)
        self._tick()

    # ── canvas size (always live) ─────────────────────────────────────────────
    def _canvas_size(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w > 10 and h > 10:
            self._cw, self._ch = w, h
        return self._cw, self._ch

    def _canvas_centre(self):
        w, h = self._canvas_size()
        return w / 2, h / 2

    # ── initial node placement ────────────────────────────────────────────────
    def _initial_placement(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 20 or h < 20:
            self.root.after(60, self._initial_placement)
            return
        self._cw, self._ch = w, h
        self._ring_placement(w, h)

    def _ring_placement(self, w, h):
        cx, cy = w / 2, h / 2
        n = len(self.graph.nodes)
        if n == 0:
            return
        r = min(w, h) * 0.28
        for i, node in enumerate(self.graph.nodes):
            ang = 2 * math.pi * i / n - math.pi / 2
            node.x  = cx + r * math.cos(ang)
            node.y  = cy + r * math.sin(ang)
            node.vx = node.vy = 0.0

    # ── canvas button (colour-correct on macOS) ───────────────────────────────
    def _make_btn(self, parent, text, cmd,
                  fg="#fff", bg="#2d2d44", hov="#3d3d58"):
        fr = tk.Frame(parent, bg=PANEL, pady=6, padx=3)
        fr.pack(side=tk.LEFT)
        # measure text
        tmp = tk.Label(fr, text=text, font=("Helvetica", 11, "bold"))
        tmp.update_idletasks()
        tw, th = tmp.winfo_reqwidth(), tmp.winfo_reqheight()
        tmp.destroy()
        px, py = 16, 7
        W, H, rad = tw + px*2, th + py*2, 7

        cv = tk.Canvas(fr, width=W, height=H, bg=PANEL,
                       highlightthickness=0, cursor="hand2")
        cv.pack()

        def _pill(col):
            cv.delete("all")
            for x0, y0, x1, y1, s in [
                (0, 0, rad*2, rad*2, 90), (W-rad*2, 0, W, rad*2, 0),
                (0, H-rad*2, rad*2, H, 180), (W-rad*2, H-rad*2, W, H, 270),
            ]:
                cv.create_arc(x0, y0, x1, y1, start=s, extent=90,
                              fill=col, outline=col)
            cv.create_rectangle(rad, 0, W-rad, H, fill=col, outline=col)
            cv.create_rectangle(0, rad, W, H-rad, fill=col, outline=col)
            cv.create_text(W//2, H//2, text=text, fill=fg,
                           font=("Helvetica", 11, "bold"))

        _pill(bg)
        cv.bind("<Enter>",           lambda e: _pill(hov))
        cv.bind("<Leave>",           lambda e: _pill(bg))
        cv.bind("<ButtonPress-1>",   lambda e: _pill(_darken(bg, 0.15)))
        cv.bind("<ButtonRelease-1>", lambda e: (_pill(hov), cmd()))
        return cv, lambda col: _pill(col)

    # ── UI build ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.root.title(f"Scene Graph Editor  ·  {self.scene_id}")
        self.root.configure(bg=BG)
        self.root.bind("<Escape>", lambda e: self._cancel_edge_mode())

        # ── toolbar ───────────────────────────────────────────────────────────
        tb = tk.Frame(self.root, bg=PANEL)
        tb.pack(fill=tk.X)

        # logo
        lf = tk.Frame(tb, bg=PANEL, padx=14, pady=7)
        lf.pack(side=tk.LEFT)
        tk.Label(lf, text="⬡", bg=PANEL, fg=ACCENT,
                 font=("Helvetica", 18, "bold")).pack(side=tk.LEFT)
        tk.Label(lf, text="  Scene Graph", bg=PANEL, fg=TEXT,
                 font=("Helvetica", 13, "bold")).pack(side=tk.LEFT)
        tk.Frame(tb, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y, pady=4)

        # buttons
        self._btns = {}
        specs = [
            ("add_node", "＋  Node",     self._dlg_add_node,
             "#fff",    ACCENT,     _darken(ACCENT)),
            ("add_edge", "⟶  Edge",     self._toggle_edge_mode,
             "#ddd",    "#2d2d44",  "#3d3d58"),
            ("del_node", "✕  Node",     self._del_sel_node,
             "#f87171", "#3a1818",  "#4a2424"),
            ("del_edge", "✕  Edge",     self._del_sel_edge,
             "#f87171", "#3a1818",  "#4a2424"),
            ("export",   "↑  Export",   self._export,
             "#a3e635", "#1a2a0a",  "#253a10"),
            ("load",     "⇥  Load JSON",self._load_json,
             "#94a3b8", "#1e2a38",  "#2a3a48"),
            ("showjson", "{ }  JSON",   self._show_json,
             "#94a3b8", "#1e2a38",  "#2a3a48"),
        ]
        for key, label, cmd, fg, bg_, hov in specs:
            w, rd = self._make_btn(tb, label, cmd, fg=fg, bg=bg_, hov=hov)
            self._btns[key] = (w, rd)

        self.mode_lbl = tk.Label(tb, text="", bg=PANEL, fg="#f97316",
                                 font=("Helvetica", 11, "bold"))
        self.mode_lbl.pack(side=tk.LEFT, padx=10)

        self.stats_lbl = tk.Label(tb, text="", bg=PANEL, fg=SUBTEXT,
                                  font=("Helvetica", 11))
        self.stats_lbl.pack(side=tk.RIGHT, padx=16)

        # ── body ──────────────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True)

        # canvas
        self.canvas = tk.Canvas(body, bg=BG, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Double-Button-1>", self._on_dblclick)
        self.canvas.bind("<ButtonPress-3>",   self._on_rclick)

        # ── sidebar ───────────────────────────────────────────────────────────
        side = tk.Frame(body, bg=PANEL, width=290)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)

        # nodes section
        nf = tk.Frame(side, bg=PANEL, padx=14, pady=10)
        nf.pack(fill=tk.X)
        tk.Label(nf, text="NODES", bg=PANEL, fg=SUBTEXT,
                 font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(0, 4))
        self.node_lb = tk.Listbox(nf, bg=CARD, fg=TEXT,
            selectbackground=ACCENT, selectforeground="#fff",
            relief=tk.FLAT, bd=0, highlightthickness=0,
            font=("Helvetica", 12), activestyle="none",
            height=7, cursor="hand2")
        self.node_lb.pack(fill=tk.X)
        self.node_lb.bind("<<ListboxSelect>>", self._on_node_lb_sel)

        tk.Frame(side, bg=BORDER, height=1).pack(fill=tk.X, padx=14, pady=6)

        # edges section
        eh = tk.Frame(side, bg=PANEL, padx=14)
        eh.pack(fill=tk.X)
        tk.Label(eh, text="EDGES", bg=PANEL, fg=SUBTEXT,
                 font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        self.edge_cnt_lbl = tk.Label(eh, text="", bg=PANEL, fg=SUBTEXT,
                                     font=("Helvetica", 10))
        self.edge_cnt_lbl.pack(side=tk.RIGHT)

        # filter
        ff = tk.Frame(side, bg=PANEL, padx=14, pady=4)
        ff.pack(fill=tk.X)
        self._fvar = tk.StringVar()
        self._fentry = tk.Entry(ff, textvariable=self._fvar,
                                bg=CARD, fg=TEXT, insertbackground=TEXT,
                                relief=tk.FLAT, bd=0, highlightthickness=1,
                                highlightcolor=ACCENT, highlightbackground=BORDER,
                                font=("Helvetica", 11))
        self._fentry.pack(fill=tk.X, ipady=5)
        self._fentry.insert(0, "filter edges…")
        self._fentry.bind("<FocusIn>",
            lambda e: self._fentry.delete(0, tk.END)
            if self._fentry.get() == "filter edges…" else None)
        self._fentry.bind("<FocusOut>",
            lambda e: self._fentry.insert(0, "filter edges…")
            if self._fentry.get() == "" else None)

        # edge listbox — keep a parallel list of displayed edges
        self._vis_edges: list[Edge] = []
        ef = tk.Frame(side, bg=PANEL, padx=14, pady=4)
        ef.pack(fill=tk.BOTH, expand=True)
        esb = tk.Scrollbar(ef, troughcolor=CARD, relief=tk.FLAT, width=6)
        esb.pack(side=tk.RIGHT, fill=tk.Y)
        self.edge_lb = tk.Listbox(ef, bg=CARD, fg=TEXT,
            selectbackground=SEL_EDGE_COL, selectforeground="#fff",
            relief=tk.FLAT, bd=0, highlightthickness=0,
            font=("Helvetica", 11), activestyle="none",
            yscrollcommand=esb.set, cursor="hand2")
        self.edge_lb.pack(fill=tk.BOTH, expand=True)
        esb.config(command=self.edge_lb.yview)
        self.edge_lb.bind("<<ListboxSelect>>", self._on_edge_lb_sel)

        # register filter trace AFTER edge_lb exists
        self._fvar.trace_add("write", lambda *_: self._refresh_lists())

        tk.Frame(side, bg=BORDER, height=1).pack(fill=tk.X, padx=14, pady=6)
        tk.Label(side,
                 text="Drag nodes freely  ·  Dbl-click to rename\n"
                      "Right-click for menu  ·  Esc to cancel edge mode",
                 bg=PANEL, fg=SUBTEXT, font=("Helvetica", 10),
                 justify=tk.LEFT, padx=14, pady=6).pack(anchor=tk.W)

        self._refresh_lists()

    # ── Physics ───────────────────────────────────────────────────────────────
    def _tick(self):
        if not self.running:
            return
        self._step_physics()
        self._render()
        self.root.after(TICK_MS, self._tick)

    def _step_physics(self):
        nodes = self.graph.nodes
        if not nodes:
            return
        W, H = self._canvas_size()
        cx, cy = W / 2, H / 2

        fx = {n: 0.0 for n in nodes}
        fy = {n: 0.0 for n in nodes}

        # ── two-zone centre force ──────────────────────────────────────────
        # Outside CORE_DEAD: inward gravity  → pulls system back to centre
        # Inside  CORE_DEAD: outward push    → nodes don't pile on the glow
        for n in nodes:
            if n.pinned:
                continue
            dx, dy = n.x - cx, n.y - cy
            d = math.hypot(dx, dy) + 1e-6
            if d < CORE_DEAD:
                # push OUT — inverse-square repulsion from core
                mag = min(CORE_REPULSE / (d * d), 80)
                fx[n] += mag * dx / d
                fy[n] += mag * dy / d
            else:
                # pull IN — linear gravity toward centre
                mag = GRAVITY_K * d
                fx[n] -= mag * dx / d
                fy[n] -= mag * dy / d

        # node–node force: repel when closer than PAIR_REST, attract when farther
        for i, a in enumerate(nodes):
            for b in nodes[i+1:]:
                dx, dy = a.x - b.x, a.y - b.y
                d = math.hypot(dx, dy) + 1e-6
                if d > 600:
                    continue
                if d < PAIR_REST:
                    mag = min(PAIR_REPULSE / (d * d), 160)   # steep push apart
                else:
                    mag = -min(PAIR_K * (d - PAIR_REST), PAIR_MAX)  # gentle pull
                fx[a] += mag * dx / d;  fy[a] += mag * dy / d
                fx[b] -= mag * dx / d;  fy[b] -= mag * dy / d

        # edge springs (attract connected nodes)
        for e in self.graph.edges:
            a, b = e.src, e.dst
            if a is b:
                continue
            dx, dy = b.x - a.x, b.y - a.y
            d = math.hypot(dx, dy) + 1e-6
            stretch = d - SPRING_LEN
            mag = SPRING_K * stretch
            fx[a] += mag * dx / d;  fy[a] += mag * dy / d
            fx[b] -= mag * dx / d;  fy[b] -= mag * dy / d

        # integrate
        pad = NODE_R + 10
        for n in nodes:
            if n.pinned:
                continue
            n.vx = (n.vx + fx[n] * TIMESTEP) * DAMPING
            n.vy = (n.vy + fy[n] * TIMESTEP) * DAMPING
            sp = math.hypot(n.vx, n.vy)
            if sp > 20:
                n.vx *= 20 / sp
                n.vy *= 20 / sp
            n.x = max(pad, min(W - pad, n.x + n.vx * TIMESTEP))
            n.y = max(pad, min(H - pad, n.y + n.vy * TIMESTEP))

    # ── Rendering ─────────────────────────────────────────────────────────────
    def _render(self):
        c = self.canvas
        c.delete("all")
        W, H = self._canvas_size()
        cx, cy = W / 2, H / 2

        # dot grid
        for xi in range(0, W, 50):
            for yi in range(0, H, 50):
                c.create_oval(xi-1, yi-1, xi+1, yi+1, fill="#1a1a28", outline="")

        # centre glow (the negative-gravity core)
        for r, col, oc in [
            (100, "#0f0f1e", "#1a1a30"),
            (65,  "#141428", "#1e1e3a"),
            (38,  "#1a1a35", "#252550"),
            (18,  "#202048", "#303060"),
        ]:
            c.create_oval(cx-r, cy-r, cx+r, cy+r, fill=col, outline=oc, width=1)
        c.create_text(cx, cy, text="⊙", fill="#3a3a80", font=("Helvetica", 13))

        # edges
        for edge in self.graph.edges:
            self._render_edge(edge)

        # rubber-band line while drawing edge
        if self.edge_mode and self.edge_src:
            c.create_line(self.edge_src.x, self.edge_src.y,
                          self._mx, self._my,
                          fill="#f97316", width=2, dash=(8, 4))
            c.create_oval(self._mx-5, self._my-5,
                          self._mx+5, self._my+5,
                          fill="#f97316", outline="")

        # nodes (on top)
        for node in self.graph.nodes:
            self._render_node(node)

        # stats
        self.stats_lbl.config(
            text=f"{len(self.graph.nodes)} nodes  ·  {len(self.graph.edges)} edges")

    def _render_edge(self, edge: Edge):
        c = self.canvas
        a, b = edge.src, edge.dst
        hi  = (edge is self.sel_edge)
        col = SEL_EDGE_COL if hi else EDGE_COL
        lw  = 2.5 if hi else 1.5

        # self-loop
        if a is b:
            r = NODE_R + 16
            c.create_oval(a.x, a.y - r*1.9, a.x + r*1.8, a.y,
                          outline=col, width=lw)
            if edge.label:
                c.create_text(a.x + r*0.9, a.y - r,
                              text=edge.label, fill=EDGE_LABEL_COL,
                              font=("Helvetica", 8, "bold"))
            return

        dx, dy = b.x - a.x, b.y - a.y
        d = math.hypot(dx, dy) + 1e-6
        sx = a.x + dx/d * (NODE_R + 3)
        sy = a.y + dy/d * (NODE_R + 3)
        ex = b.x - dx/d * (NODE_R + 7)
        ey = b.y - dy/d * (NODE_R + 7)
        # slight perpendicular curve
        perp = 20
        mx = (sx+ex)/2 - dy/d * perp
        my = (sy+ey)/2 + dx/d * perp

        c.create_line(sx, sy, mx, my, ex, ey,
                      fill=col, width=lw, smooth=True,
                      arrow=tk.LAST, arrowshape=(12, 15, 5))

        if edge.label:
            lx = (sx + 2*mx + ex) / 4
            ly = (sy + 2*my + ey) / 4
            c.create_text(lx, ly, text=edge.label,
                          fill=EDGE_LABEL_COL, font=("Helvetica", 8, "bold"))

    def _render_node(self, node: Node):
        c = self.canvas
        x, y = node.x, node.y
        r    = NODE_R
        col  = node.colour
        sel  = (node is self.sel_node) or (node is self.edge_src)

        # shadow
        c.create_oval(x-r+3, y-r+4, x+r+3, y+r+4, fill="#08080f", outline="")

        # selection glow
        if sel:
            c.create_oval(x-r-12, y-r-12, x+r+12, y+r+12,
                          fill=_lighten(col, 0.08), outline="")
            c.create_oval(x-r-6, y-r-6, x+r+6, y+r+6,
                          fill=_lighten(col, 0.22), outline="")

        # body (outer darker ring + inner full colour)
        c.create_oval(x-r, y-r, x+r, y+r, fill=_darken(col), outline="")
        ir = int(r * 0.80)
        c.create_oval(x-ir, y-ir, x+ir, y+ir, fill=col, outline="")

        # border
        bord = "#ffffff" if sel else _lighten(col, 0.55)
        c.create_oval(x-r, y-r, x+r, y+r,
                      fill="", outline=bord, width=3 if sel else 1.8)

        # label: name line + id line
        c.create_text(x, y-7, text=node.label, fill="#fff",
                      font=("Helvetica", 10, "bold"), width=r*2-4)
        c.create_text(x, y+10, text=f"({node.id})",
                      fill=_lighten(col, 0.75), font=("Helvetica", 9))

    # ── hit-testing ───────────────────────────────────────────────────────────
    def _node_at(self, x, y) -> "Node | None":
        for n in reversed(self.graph.nodes):
            if math.hypot(n.x - x, n.y - y) <= NODE_R + 5:
                return n
        return None

    def _edge_near(self, x, y, tol=10) -> "Edge | None":
        for e in self.graph.edges:
            a, b = e.src, e.dst
            if a is b:
                continue
            dx, dy = b.x - a.x, b.y - a.y
            t = ((x-a.x)*dx + (y-a.y)*dy) / (dx*dx + dy*dy + 1e-9)
            t = max(0.0, min(1.0, t))
            if math.hypot(a.x + t*dx - x, a.y + t*dy - y) < tol:
                return e
        return None

    # ── canvas events ─────────────────────────────────────────────────────────
    def _on_press(self, ev):
        x, y = ev.x, ev.y
        n = self._node_at(x, y)

        if self.edge_mode:
            if n is None:
                # clicked empty space — cancel
                self._cancel_edge_mode()
                return
            if self.edge_src is None:
                # first click: set source
                self.edge_src = n
                self.mode_lbl.config(
                    text=f"⟶  now click TARGET  (src: {n.label})")
            else:
                # second click: open edge dialog
                self._open_edge_dialog(self.edge_src, n)
                self._cancel_edge_mode()
            return

        # normal mode
        if n:
            self.sel_node  = n
            self.sel_edge  = None
            self.drag_node = n
            self.drag_ox   = x - n.x
            self.drag_oy   = y - n.y
            n.pinned = True
        else:
            e = self._edge_near(x, y)
            self.sel_edge  = e
            self.sel_node  = None
        self._refresh_lists()

    def _on_drag(self, ev):
        self._mx, self._my = ev.x, ev.y
        if self.drag_node:
            self.drag_node.x = ev.x - self.drag_ox
            self.drag_node.y = ev.y - self.drag_oy

    def _on_release(self, ev):
        if self.drag_node:
            self.drag_node.pinned = False
            self.drag_node.vx = self.drag_node.vy = 0.0
            self.drag_node = None

    def _on_dblclick(self, ev):
        n = self._node_at(ev.x, ev.y)
        if n:
            self._dlg_rename_node(n)

    def _on_rclick(self, ev):
        n = self._node_at(ev.x, ev.y)
        e = None if n else self._edge_near(ev.x, ev.y)
        m = tk.Menu(self.root, tearoff=0, bg=CARD, fg=TEXT,
                    activebackground=ACCENT, activeforeground="#fff",
                    relief=tk.FLAT, bd=0)
        if n:
            m.add_command(label=f'Rename  "{n.label}"',
                          command=lambda: self._dlg_rename_node(n))
            m.add_command(label="Draw edge from here",
                          command=lambda: self._start_edge_from(n))
            m.add_separator()
            m.add_command(label=f'Delete  "{n.label}"',
                          command=lambda: self._delete_node(n))
        elif e:
            m.add_command(label=f'Edit label  "{e.label}"',
                          command=lambda: self._dlg_rename_edge(e))
            m.add_separator()
            m.add_command(label="Delete this edge",
                          command=lambda: self._delete_edge(e))
        else:
            m.add_command(label="Add node here",
                          command=lambda: self._dlg_add_node(ev.x, ev.y))
        m.post(ev.x_root, ev.y_root)

    # ── edge mode ─────────────────────────────────────────────────────────────
    def _toggle_edge_mode(self):
        if self.edge_mode:
            self._cancel_edge_mode()
        else:
            self.edge_mode = True
            self.edge_src  = None
            self._btns["add_edge"][1]("#f97316")
            self.mode_lbl.config(text="⟶  EDGE MODE  —  click SOURCE node  (Esc to cancel)")

    def _cancel_edge_mode(self):
        self.edge_mode = False
        self.edge_src  = None
        self._btns["add_edge"][1]("#2d2d44")
        self.mode_lbl.config(text="")

    def _start_edge_from(self, n: Node):
        self.edge_mode = True
        self.edge_src  = n
        self._btns["add_edge"][1]("#f97316")
        self.mode_lbl.config(text=f"⟶  now click TARGET  (src: {n.label})  (Esc to cancel)")

    # ── dialogs ───────────────────────────────────────────────────────────────
    def _open_edge_dialog(self, src: Node, dst: Node):
        win = tk.Toplevel(self.root)
        win.title("Add Relationship")
        win.configure(bg=CARD)
        win.grab_set()
        win.resizable(False, False)

        tk.Label(win,
                 text=f"{src.label} ({src.id})  ⟶  {dst.label} ({dst.id})",
                 bg=CARD, fg=TEXT, font=("Helvetica", 12, "bold"),
                 padx=24, pady=16).pack()

        inner = tk.Frame(win, bg=CARD, padx=24)
        inner.pack(fill=tk.X)
        tk.Label(inner, text="Relationship type:", bg=CARD, fg=SUBTEXT,
                 font=("Helvetica", 10)).pack(anchor=tk.W)

        var = tk.StringVar(value=SPATIAL_RELS[0])
        cb = ttk.Combobox(inner, textvariable=var,
                          values=ALL_RELS, width=28,
                          font=("Helvetica", 11))
        cb.pack(pady=8)
        cb.focus_set()

        # also allow free-text custom label
        tk.Label(inner, text="Or type a custom label:",
                 bg=CARD, fg=SUBTEXT, font=("Helvetica", 10)).pack(anchor=tk.W)
        custom_var = tk.StringVar()
        custom_e = tk.Entry(inner, textvariable=custom_var,
                            bg="#252538", fg=TEXT, insertbackground=TEXT,
                            relief=tk.FLAT, highlightthickness=1,
                            highlightbackground=BORDER,
                            font=("Helvetica", 11))
        custom_e.pack(fill=tk.X, ipady=4, pady=(4, 12))

        def ok():
            label = custom_var.get().strip() or var.get().strip()
            if label:
                self.graph.add_edge(src, dst, label)
                self._refresh_lists()
            win.destroy()

        btn_f = tk.Frame(win, bg=CARD, pady=12)
        btn_f.pack()
        # use canvas button so it's coloured on macOS
        _, _ = self._make_btn(btn_f, "  Add Edge  ", ok,
                              fg="#fff", bg=ACCENT, hov=_darken(ACCENT))
        win.bind("<Return>", lambda _: ok())

    def _dlg_add_node(self, x=None, y=None):
        label = simpledialog.askstring(
            "Add Node", "Object label (e.g. chair, table, lamp):",
            parent=self.root)
        if not label:
            return
        W, H = self._canvas_size()
        nx_ = x if x is not None else random.uniform(W*0.35, W*0.65)
        ny_ = y if y is not None else random.uniform(H*0.35, H*0.65)
        self.graph.add_node(label.strip(), x=nx_, y=ny_)
        self._refresh_lists()

    def _dlg_rename_node(self, n: Node):
        v = simpledialog.askstring(
            "Rename Node", f"New label for '{n.label}':",
            initialvalue=n.label, parent=self.root)
        if v:
            n.label  = v.strip()
            n.colour = node_col(n.label)
            self._refresh_lists()

    def _dlg_rename_edge(self, e: Edge):
        v = simpledialog.askstring(
            "Edit Edge Label",
            f"Label for:  {e.src.label} → {e.dst.label}",
            initialvalue=e.label, parent=self.root)
        if v is not None:
            e.label = v.strip()
            self._refresh_lists()

    def _del_sel_node(self):
        if self.sel_node:
            self._delete_node(self.sel_node)

    def _delete_node(self, n: Node):
        if messagebox.askyesno(
                "Delete Node",
                f"Remove '{n.label} ({n.id})' and all its edges?",
                parent=self.root):
            self.graph.remove_node(n)
            if self.sel_node is n:
                self.sel_node = None
            self._refresh_lists()

    def _del_sel_edge(self):
        if self.sel_edge:
            self._delete_edge(self.sel_edge)

    def _delete_edge(self, e: Edge):
        self.graph.remove_edge(e)
        if self.sel_edge is e:
            self.sel_edge = None
        self._refresh_lists()

    def _export(self):
        path = filedialog.asksaveasfilename(
            parent=self.root,
            initialdir=os.path.dirname(self.out_png),
            initialfile=os.path.basename(self.out_png),
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
            title="Export Scene Graph PNG")
        if path:
            export_png(self.graph, path, f"Scene Graph: {self.scene_id}")
            messagebox.showinfo("Exported", f"Saved to:\n{path}", parent=self.root)

    def _load_json(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Load relationships JSON",
            initialdir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not path:
            return
        sid = simpledialog.askstring("Scene ID", "Enter scan ID to load:",
                                     parent=self.root)
        if not sid:
            return
        g = load_scene(path, sid.strip())
        if g is None:
            messagebox.showerror("Not found",
                f"Scene '{sid}' not found in:\n{path}", parent=self.root)
            return
        self.graph    = g
        self.scene_id = sid.strip()
        self.sel_node = self.sel_edge = None
        self.root.title(f"Scene Graph Editor  ·  {self.scene_id}")
        W, H = self._canvas_size()
        self._ring_placement(W, H)
        self._refresh_lists()

    def _show_json(self):
        text = json.dumps(self.graph.to_dict(), indent=2)
        win = tk.Toplevel(self.root)
        win.title("Scene Graph JSON")
        win.configure(bg=CARD)
        win.geometry("540x480")
        fr = tk.Frame(win, bg=CARD, padx=8, pady=8)
        fr.pack(fill=tk.BOTH, expand=True)
        vsb = tk.Scrollbar(fr); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb = tk.Scrollbar(fr, orient=tk.HORIZONTAL); hsb.pack(side=tk.BOTTOM, fill=tk.X)
        tx = tk.Text(fr, bg="#0d0d1a", fg="#a6e3a1",
                     font=("Courier New", 10),
                     yscrollcommand=vsb.set, xscrollcommand=hsb.set,
                     wrap=tk.NONE, relief=tk.FLAT)
        tx.insert(tk.END, text)
        tx.config(state=tk.DISABLED)
        tx.pack(fill=tk.BOTH, expand=True)
        vsb.config(command=tx.yview)
        hsb.config(command=tx.xview)
        bf = tk.Frame(win, bg=CARD, pady=8); bf.pack()
        def _copy():
            win.clipboard_clear(); win.clipboard_append(text)
        tk.Button(bf, text="Copy to clipboard", command=_copy,
                  bg=ACCENT, fg="#fff", relief=tk.FLAT,
                  padx=12, pady=5, font=("Helvetica", 10)).pack(side=tk.LEFT, padx=6)
        tk.Button(bf, text="Close", command=win.destroy,
                  bg=BORDER, fg=TEXT, relief=tk.FLAT,
                  padx=12, pady=5, font=("Helvetica", 10)).pack(side=tk.LEFT)

    # ── sidebar refresh ───────────────────────────────────────────────────────
    def _refresh_lists(self):
        # nodes
        self.node_lb.delete(0, tk.END)
        for n in self.graph.nodes:
            self.node_lb.insert(tk.END, f"  {n.label}  ({n.id})")
            if n is self.sel_node:
                self.node_lb.itemconfig(tk.END, fg=ACCENT)

        # edges — keep _vis_edges in sync so index → Edge is correct
        filt = self._fvar.get().strip().lower()
        if filt in ("filter edges…", ""):
            filt = ""
        self._vis_edges = [
            e for e in self.graph.edges
            if not filt
            or filt in e.label.lower()
            or filt in e.src.label.lower()
            or filt in e.dst.label.lower()
        ]

        self.edge_lb.delete(0, tk.END)
        for e in self._vis_edges:
            row = f"  {e.src.label}({e.src.id}) → {e.dst.label}({e.dst.id})"
            if e.label:
                row += f"  [{e.label}]"
            self.edge_lb.insert(tk.END, row)
            if e is self.sel_edge:
                self.edge_lb.itemconfig(tk.END, fg=SEL_EDGE_COL)

        self.edge_cnt_lbl.config(
            text=f"{len(self._vis_edges)}/{len(self.graph.edges)}")

    def _on_node_lb_sel(self, _):
        sel = self.node_lb.curselection()
        if sel:
            idx = sel[0]
            if idx < len(self.graph.nodes):
                self.sel_node = self.graph.nodes[idx]
                self.sel_edge = None

    def _on_edge_lb_sel(self, _):
        sel = self.edge_lb.curselection()
        if sel:
            idx = sel[0]
            if idx < len(self._vis_edges):   # use filtered list, not full list
                self.sel_edge = self._vis_edges[idx]
                self.sel_node = None

    def on_close(self):
        self.running = False
        self.root.destroy()


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _DATA = os.path.join(_HERE, "data")
    _OUT  = os.path.join(_HERE, "output")
    os.makedirs(_DATA, exist_ok=True)
    os.makedirs(_OUT,  exist_ok=True)

    ap = argparse.ArgumentParser(description="Scene Graph Editor")
    ap.add_argument("--scene", default="DiningRoom-31158")
    ap.add_argument("--rel",   default=os.path.join(
                                   _DATA, "relationships_diningroom_test.json"))
    ap.add_argument("--out",   default=os.path.join(
                                   _OUT,  "DiningRoom-31158_scene_graph.png"))
    ap.add_argument("--all-edges", action="store_true",
                    help="Load all relationship types (default: spatial only)")
    args = ap.parse_args()

    graph = load_scene(args.rel, args.scene,
                       spatial_only=not args.all_edges)
    if graph is None:
        print(f"[warn] '{args.scene}' not found — using demo graph.")
        graph    = demo_graph()
        scene_id = "Demo"
    else:
        scene_id = args.scene
        print(f"[info] Loaded {len(graph.nodes)} nodes, "
              f"{len(graph.edges)} edges  ({scene_id})")

    root = tk.Tk()
    root.geometry("1300x820")

    app = App(root, graph, scene_id, args.out)
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    if HAS_MPL:
        def _bg_export():
            time.sleep(2.5)
            export_png(graph, args.out, f"Scene Graph: {scene_id}")
        threading.Thread(target=_bg_export, daemon=True).start()

    root.mainloop()


if __name__ == "__main__":
    main()
