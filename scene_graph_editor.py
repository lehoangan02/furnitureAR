"""
Scene Graph Editor  —  furnitureAR
===================================
Interactive physics-based scene-graph editor for 3D-FRONT / EchoScene data.

Controls
--------
  Drag node          : left-click + drag
  Double-click node  : rename
  Right-click        : context menu (add/rename/delete node or edge)
  + Add Node button  : add new object node
  + Add Edge button  : click source then target on canvas
  Del Node / Del Edge: delete selected item

Run
---
  python scene_graph_editor.py
"""

import argparse, json, math, os, random, sys, threading, time
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
    "bed": "#4a9eff", "chair": "#e74c3c", "dining_chair": "#e74c3c",
    "nightstand": "#a855f7", "table": "#f97316", "dining_table": "#f97316",
    "wardrobe": "#22c55e", "lamp": "#eab308", "pendant_lamp": "#eab308",
    "floor": "#94a3b8", "sofa": "#14b8a6", "desk": "#f97316",
    "cabinet": "#7c3aed", "bookshelf": "#16a34a", "tv": "#3b82f6",
    "monitor": "#3b82f6", "stool": "#dc2626",
}
DEFAULT_COLOUR  = "#06b6d4"
EDGE_COLOUR     = "#475569"
EDGE_LABEL_COL  = "#f87171"
SEL_EDGE_COLOUR = "#fb923c"

SPATIAL_RELS = {
    "left", "right", "front", "behind", "above",
    "close by", "standing on", "below", "opposite to",
}

# ── physics ───────────────────────────────────────────────────────────────────
SPRING_LEN   = 220
SPRING_K     = 0.035
REPULSE_K    = 9000
CENTER_K     = 18000
CENTER_DEAD  = 80    # px — centre dead-zone radius (no push inside)
DAMPING      = 0.78
TIMESTEP     = 0.6
NODE_R       = 38
TICK_MS      = 28


# ── helpers ───────────────────────────────────────────────────────────────────
def _hex(h): h=h.lstrip("#"); return tuple(int(h[i:i+2],16) for i in (0,2,4))
def _lighten(c, f=0.45):
    r,g,b = _hex(c)
    return f"#{int(r+(255-r)*f):02x}{int(g+(255-g)*f):02x}{int(b+(255-b)*f):02x}"
def _darken(c, f=0.3):
    r,g,b = _hex(c)
    return f"#{int(r*(1-f)):02x}{int(g*(1-f)):02x}{int(b*(1-f)):02x}"

def node_col(label: str) -> str:
    l = label.lower()
    for k, v in COLOUR_MAP.items():
        if k in l: return v
    return DEFAULT_COLOUR


# ── data model ────────────────────────────────────────────────────────────────
class Node:
    _counter = 0
    def __init__(self, label, nid=None, x=None, y=None):
        if nid is None:
            Node._counter += 1; nid = Node._counter
        self.id, self.label = nid, label
        self.colour = node_col(label)
        self.x  = x or 0.0;  self.y  = y or 0.0
        self.vx = 0.0;        self.vy = 0.0
        self.pinned = False
    @property
    def display(self): return f"{self.label}\n({self.id})"
    @property
    def short(self):   return f"{self.label} ({self.id})"


class Edge:
    def __init__(self, src, dst, label=""):
        self.src, self.dst, self.label = src, dst, label


class SceneGraph:
    def __init__(self):
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self._map:  dict[int, Node] = {}

    def add_node(self, label, nid=None, x=None, y=None):
        n = Node(label, nid, x, y)
        while n.id in self._map: n.id += 1000
        self.nodes.append(n); self._map[n.id] = n
        if n.id > Node._counter: Node._counter = n.id
        return n

    def remove_node(self, n):
        self.nodes = [x for x in self.nodes if x is not n]
        self.edges = [e for e in self.edges if e.src is not n and e.dst is not n]
        self._map.pop(n.id, None)

    def add_edge(self, src, dst, label=""):
        e = Edge(src, dst, label); self.edges.append(e); return e

    def remove_edge(self, e):
        self.edges = [x for x in self.edges if x is not e]

    def node_by_id(self, nid): return self._map.get(nid)

    def to_dict(self):
        return {
            "objects": {str(n.id): n.label for n in self.nodes},
            "relationships": [[e.src.id, e.dst.id, i, e.label]
                              for i, e in enumerate(self.edges)],
        }


# ── loader ────────────────────────────────────────────────────────────────────
def load_scene(rel_json, scene_id, spatial_only=True):
    if not os.path.exists(rel_json): return None
    with open(rel_json) as f: data = json.load(f)
    scan = next((s for s in data.get("scans",[]) if s["scan"]==scene_id), None)
    if not scan: return None

    g = SceneGraph()
    objs = scan["objects"]
    n = len(objs)
    # place in a ring — we'll re-centre once canvas is known
    for i, (k, v) in enumerate(objs.items()):
        ang = 2*math.pi*i/max(n,1)
        g.add_node(v, int(k), math.cos(ang)*200, math.sin(ang)*200)

    for rel in scan.get("relationships", []):
        src_id, dst_id, _, rel_type = rel
        if spatial_only and rel_type not in SPATIAL_RELS: continue
        src = g.node_by_id(src_id); dst = g.node_by_id(dst_id)
        if src and dst:
            if rel_type == "standing on" and dst.label == "floor": continue
            g.add_edge(src, dst, rel_type)
    return g


def demo_graph():
    g = SceneGraph()
    t = g.add_node("dining_table",  1)
    c1= g.add_node("dining_chair",  2); c2= g.add_node("dining_chair", 3)
    c3= g.add_node("dining_chair",  4); c4= g.add_node("dining_chair", 5)
    lp= g.add_node("pendant_lamp",  6); fl= g.add_node("floor",        7)
    g.add_edge(lp, t,  "above"); g.add_edge(lp, c1, "above")
    g.add_edge(c1, t,  "left"); g.add_edge(c2, t, "right")
    g.add_edge(c3, c4, "behind"); g.add_edge(c2, c4, "close by")
    g.add_edge(t,  fl, "standing on")
    return g


# ── PNG export ────────────────────────────────────────────────────────────────
def export_png(graph, out_path, title="Scene Graph"):
    if not HAS_MPL:
        messagebox.showerror("Missing","pip install matplotlib networkx"); return
    G = nx.DiGraph()
    for n in graph.nodes: G.add_node(n.id)
    for e in graph.edges:
        if G.has_edge(e.src.id, e.dst.id):
            G[e.src.id][e.dst.id]["label"] += f", {e.label}"
        else:
            G.add_edge(e.src.id, e.dst.id, label=e.label)

    labels  = {n.id: f"{n.label} ({n.id})" for n in graph.nodes}
    colours = [node_col(n.label) for n in graph.nodes]
    pos     = {n.id: (n.x, -n.y) for n in graph.nodes}

    fig, ax = plt.subplots(figsize=(12,9))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=3200,
        node_color=colours, edgecolors="black", linewidths=1.8)
    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels,
        font_size=9, font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.6,
        arrowstyle="-|>", arrowsize=22, edge_color=EDGE_COLOUR,
        connectionstyle="arc3,rad=0.06")
    nx.draw_networkx_edge_labels(G, pos, ax=ax,
        edge_labels=nx.get_edge_attributes(G,"label"),
        font_size=7.5, font_color=EDGE_LABEL_COL, font_weight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.axis("off"); plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[export] → {out_path}")


# ── main app ──────────────────────────────────────────────────────────────────
BG      = "#0f0f17"
PANEL   = "#16161f"
CARD    = "#1e1e2e"
BORDER  = "#2a2a3e"
TEXT    = "#e2e8f0"
SUBTEXT = "#64748b"
ACCENT  = "#6366f1"
BTN_BG  = "#1e1e30"
BTN_HOV = "#2d2d44"

class App:
    def __init__(self, root, graph, scene_id, out_png):
        self.root      = root
        self.graph     = graph
        self.scene_id  = scene_id
        self.out_png   = out_png
        self.running   = True

        self.sel_node : Node | None = None
        self.sel_edge : Edge | None = None
        self.drag_node: Node | None = None
        self.drag_ox = self.drag_oy = 0.0

        self.edge_mode   = False
        self.edge_src    : Node | None = None
        self._mx = self._my = 0.0

        self.canvas_w = 1050
        self.canvas_h = 720
        self._centered = False   # have we placed nodes at canvas centre yet?

        self._build()
        # Delay initial ring placement until after Tk finishes its full
        # layout pass — only then does winfo_width() return the true
        # canvas-only width (excluding the sidebar).
        self.root.after(150, self._initial_placement)
        self._tick()

    # ── Custom canvas button (colours actually work on macOS) ─────────────────
    def _make_btn(self, parent, text, command, fg="#ffffff",
                  bg="#2a2a3e", hover="#3a3a54", active_bg=None, font_size=11):
        PAD_X, PAD_H = 14, 8
        f = tk.Frame(parent, bg=PANEL, pady=5, padx=2)
        f.pack(side=tk.LEFT)
        tmp = tk.Label(f, text=text, font=("Helvetica", font_size, "bold"))
        tmp.update_idletasks()
        tw = tmp.winfo_reqwidth(); th = tmp.winfo_reqheight()
        tmp.destroy()
        W = tw + PAD_X * 2; H = th + PAD_H * 2
        c = tk.Canvas(f, width=W, height=H, bg=PANEL,
                      highlightthickness=0, cursor="hand2")
        c.pack()
        act = active_bg or bg; r = 6

        def _draw(col):
            c.delete("all")
            c.create_arc(0, 0, r*2, r*2,       start=90,  extent=90, fill=col, outline=col)
            c.create_arc(W-r*2, 0, W, r*2,     start=0,   extent=90, fill=col, outline=col)
            c.create_arc(0, H-r*2, r*2, H,     start=180, extent=90, fill=col, outline=col)
            c.create_arc(W-r*2, H-r*2, W, H,   start=270, extent=90, fill=col, outline=col)
            c.create_rectangle(r, 0, W-r, H,   fill=col, outline=col)
            c.create_rectangle(0, r, W, H-r,   fill=col, outline=col)
            c.create_text(W//2, H//2, text=text, fill=fg,
                          font=("Helvetica", font_size, "bold"))

        _draw(bg)
        c.bind("<Enter>",           lambda e: _draw(hover))
        c.bind("<Leave>",           lambda e: _draw(bg))
        c.bind("<ButtonPress-1>",   lambda e: _draw(act))
        c.bind("<ButtonRelease-1>", lambda e: (_draw(hover), command()))
        return c, _draw

    # ── Build UI ──────────────────────────────────────────────────────────────
    def _build(self):
        self.root.title(f"Scene Graph Editor  ·  {self.scene_id}")
        self.root.configure(bg=BG)

        # ── toolbar ───────────────────────────────────────────────────────────
        tb = tk.Frame(self.root, bg=PANEL)
        tb.pack(fill=tk.X, side=tk.TOP)

        logo = tk.Frame(tb, bg=PANEL, padx=16, pady=8)
        logo.pack(side=tk.LEFT)
        tk.Label(logo, text="⬡", bg=PANEL, fg=ACCENT,
                 font=("Helvetica", 20, "bold")).pack(side=tk.LEFT)
        tk.Label(logo, text="  Scene Graph", bg=PANEL, fg=TEXT,
                 font=("Helvetica", 13, "bold")).pack(side=tk.LEFT)

        tk.Frame(tb, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y, pady=4)

        self._btns = {}
        btn_specs = [
            ("add_node", "＋  Node",      self._dlg_add_node,     "#fff",    ACCENT,         _darken(ACCENT, 0.2)),
            ("add_edge", "⟶  Edge",      self._toggle_edge_mode, "#cccccc", "#2d2d44",      "#3d3d58"),
            ("del_node", "✕  Node",      self._del_sel_node,     "#f87171", "#3a1a1a",      "#4a2020"),
            ("del_edge", "✕  Edge",      self._del_sel_edge,     "#f87171", "#3a1a1a",      "#4a2020"),
            ("export",   "↑  Export",    self._export,           "#a3e635", "#1a2a0a",      "#2a3a10"),
            ("load",     "⇥  Load JSON", self._load_json,        "#94a3b8", "#1e2a38",      "#2e3a48"),
            ("json",     "{ }  JSON",    self._show_json,        "#94a3b8", "#1e2a38",      "#2e3a48"),
        ]
        for key, label, cmd, fg, bg_, hov in btn_specs:
            widget, redraw = self._make_btn(tb, label, cmd, fg=fg,
                                            bg=bg_, hover=hov, font_size=11)
            self._btns[key] = (widget, redraw)

        self.mode_lbl = tk.Label(tb, text="", bg=PANEL, fg="#f97316",
                                 font=("Helvetica", 11, "bold"))
        self.mode_lbl.pack(side=tk.LEFT, padx=12)

        self.stats_lbl = tk.Label(tb, text="", bg=PANEL, fg=SUBTEXT,
                                  font=("Helvetica", 11))
        self.stats_lbl.pack(side=tk.RIGHT, padx=18)

        # ── body ──────────────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(body, bg=BG, highlightthickness=0,
                                width=self.canvas_w, height=self.canvas_h)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>",       self._on_configure)
        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Double-Button-1>", self._on_dblclick)
        self.canvas.bind("<ButtonPress-3>",   self._on_rclick)

        # ── sidebar ───────────────────────────────────────────────────────────
        side = tk.Frame(body, bg=PANEL, width=280)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)

        nf = tk.Frame(side, bg=PANEL, padx=14, pady=12)
        nf.pack(fill=tk.X)
        tk.Label(nf, text="NODES", bg=PANEL, fg=SUBTEXT,
                 font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.node_lb = tk.Listbox(nf, bg=CARD, fg=TEXT,
            selectbackground=ACCENT, selectforeground="#fff",
            relief=tk.FLAT, bd=0, highlightthickness=0,
            font=("Helvetica", 12), activestyle="none",
            height=7, cursor="hand2")
        self.node_lb.pack(fill=tk.X)
        self.node_lb.bind("<<ListboxSelect>>", self._on_node_lb)

        tk.Frame(side, bg=BORDER, height=1).pack(fill=tk.X, padx=14, pady=8)

        eh = tk.Frame(side, bg=PANEL, padx=14)
        eh.pack(fill=tk.X)
        tk.Label(eh, text="EDGES", bg=PANEL, fg=SUBTEXT,
                 font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        self.edge_count_lbl = tk.Label(eh, text="", bg=PANEL, fg=SUBTEXT,
                                       font=("Helvetica", 10))
        self.edge_count_lbl.pack(side=tk.RIGHT)

        sf = tk.Frame(side, bg=PANEL, padx=14, pady=5)
        sf.pack(fill=tk.X)
        self._filter_var = tk.StringVar()
        fentry = tk.Entry(sf, textvariable=self._filter_var,
                          bg=CARD, fg=TEXT, insertbackground=TEXT,
                          relief=tk.FLAT, bd=0, highlightthickness=1,
                          highlightcolor=ACCENT, highlightbackground=BORDER,
                          font=("Helvetica", 11))
        fentry.pack(fill=tk.X, ipady=5)
        fentry.insert(0, "filter edges…")
        fentry.bind("<FocusIn>",  lambda e: fentry.delete(0, tk.END)
                    if fentry.get() == "filter edges…" else None)
        fentry.bind("<FocusOut>", lambda e: fentry.insert(0, "filter edges…")
                    if fentry.get() == "" else None)

        el_frame = tk.Frame(side, bg=PANEL, padx=14, pady=4)
        el_frame.pack(fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(el_frame, troughcolor=CARD, relief=tk.FLAT, width=6)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.edge_lb = tk.Listbox(el_frame, bg=CARD, fg=TEXT,
            selectbackground=SEL_EDGE_COLOUR, selectforeground="#fff",
            relief=tk.FLAT, bd=0, highlightthickness=0,
            font=("Helvetica", 11), activestyle="none",
            yscrollcommand=sb.set, cursor="hand2")
        self.edge_lb.pack(fill=tk.BOTH, expand=True)
        sb.config(command=self.edge_lb.yview)
        self.edge_lb.bind("<<ListboxSelect>>", self._on_edge_lb)

        # register AFTER edge_lb exists
        self._filter_var.trace_add("write", lambda *_: self._refresh_lists())

        tk.Frame(side, bg=BORDER, height=1).pack(fill=tk.X, padx=14, pady=8)
        tk.Label(side,
                 text="Drag nodes  ·  Dbl-click to rename\n"
                      "Right-click for menu  ·  ⟶ Edge: src→dst",
                 bg=PANEL, fg=SUBTEXT, font=("Helvetica", 10),
                 justify=tk.LEFT, padx=14, pady=6).pack(anchor=tk.W)

        self._refresh_lists()

    # ── Initial placement (deferred 150 ms so layout is complete) ────────────
    def _initial_placement(self):
        """Called once after Tk finishes layout. winfo_width/height are final."""
        W = self.canvas.winfo_width()
        H = self.canvas.winfo_height()
        if W < 10 or H < 10:
            # Still not ready — retry in another 100 ms
            self.root.after(100, self._initial_placement)
            return
        self.canvas_w, self.canvas_h = W, H
        self._centered = True
        self._place_nodes_in_ring(W, H)

    # ── canvas Configure — only tracks resize, never places nodes ─────────────
    def _on_configure(self, event):
        self.canvas_w = event.width
        self.canvas_h = event.height
        # On resize after initial placement, re-centre so nodes stay inside
        if self._centered and (event.width > 10):
            self._place_nodes_in_ring(event.width, event.height)
            # Reset velocities so physics re-settles from new positions
            for n in self.graph.nodes:
                n.vx = n.vy = 0

    def _place_nodes_in_ring(self, W, H):
        cx, cy = W / 2, H / 2
        n = len(self.graph.nodes)
        r = min(W, H) * 0.30
        for i, node in enumerate(self.graph.nodes):
            ang = 2 * math.pi * i / max(n, 1) - math.pi / 2
            node.x = cx + r * math.cos(ang)
            node.y = cy + r * math.sin(ang)
            node.vx = node.vy = 0

    def _live_dims(self):
        """Always return the actual rendered canvas pixel size."""
        W = self.canvas.winfo_width()
        H = self.canvas.winfo_height()
        if W < 2 or H < 2:
            W, H = self.canvas_w, self.canvas_h
        else:
            self.canvas_w, self.canvas_h = W, H
        return W, H

    # ── Physics ───────────────────────────────────────────────────────────────
    def _tick(self):
        if not self.running: return
        self._physics()
        self._draw()
        self.root.after(TICK_MS, self._tick)

    def _physics(self):
        nodes = self.graph.nodes
        if not nodes: return
        W, H = self._live_dims()
        cx, cy = W / 2, H / 2

        fx = {n: 0.0 for n in nodes}
        fy = {n: 0.0 for n in nodes}

        # centre repulsion — push away from the real canvas centre
        for n in nodes:
            if n.pinned: continue
            dx, dy = n.x - cx, n.y - cy
            dist = math.hypot(dx, dy) + 1e-6
            if dist < CENTER_DEAD: continue
            mag = min(CENTER_K / (dist * dist), 60)
            fx[n] += mag * dx / dist
            fy[n] += mag * dy / dist

        # node–node repulsion
        for i, a in enumerate(nodes):
            for b in nodes[i+1:]:
                dx, dy = a.x - b.x, a.y - b.y
                dist = math.hypot(dx, dy) + 1e-6
                if dist > 500: continue
                mag = min(REPULSE_K / (dist * dist), 150)
                fx[a] += mag * dx / dist; fy[a] += mag * dy / dist
                fx[b] -= mag * dx / dist; fy[b] -= mag * dy / dist

        # edge springs
        for e in self.graph.edges:
            a, b = e.src, e.dst
            if a is b: continue
            dx, dy = b.x - a.x, b.y - a.y
            dist = math.hypot(dx, dy) + 1e-6
            stretch = dist - SPRING_LEN
            mag = SPRING_K * stretch
            fx[a] += mag * dx / dist; fy[a] += mag * dy / dist
            fx[b] -= mag * dx / dist; fy[b] -= mag * dy / dist

        # integrate — clamp to real canvas bounds
        pad = NODE_R + 8
        for n in nodes:
            if n.pinned: continue
            n.vx = (n.vx + fx[n] * TIMESTEP) * DAMPING
            n.vy = (n.vy + fy[n] * TIMESTEP) * DAMPING
            sp = math.hypot(n.vx, n.vy)
            if sp > 18: n.vx *= 18/sp; n.vy *= 18/sp
            n.x = max(pad, min(W - pad, n.x + n.vx * TIMESTEP))
            n.y = max(pad, min(H - pad, n.y + n.vy * TIMESTEP))

    # ── Drawing ───────────────────────────────────────────────────────────────
    def _draw(self):
        c = self.canvas
        c.delete("all")
        W, H = self._live_dims()
        cx, cy = W / 2, H / 2

        # subtle dot grid — covers full live canvas
        step = 48
        for xi in range(0, W, step):
            for yi in range(0, H, step):
                c.create_oval(xi-1, yi-1, xi+1, yi+1, fill="#1a1a28", outline="")

        # centre glow — always at the real canvas centre
        for r, col in [(90,"#12122a"),(58,"#17172e"),(32,"#1d1d38"),(14,"#232348")]:
            c.create_oval(cx-r, cy-r, cx+r, cy+r, fill=col, outline="#2a2a50", width=1)
        c.create_text(cx, cy, text="⊙", fill="#3a3a70", font=("Helvetica", 11))
        # edges
        for edge in self.graph.edges:
            self._draw_edge(edge)

        # rubber-band while drawing a new edge
        if self.edge_mode and self.edge_src:
            c.create_line(self.edge_src.x, self.edge_src.y, self._mx, self._my,
                          fill="#f97316", width=2, dash=(8, 4))
            c.create_oval(self._mx-5, self._my-5, self._mx+5, self._my+5,
                          fill="#f97316", outline="")

        # nodes on top
        for node in self.graph.nodes:
            self._draw_node(node)

        # stats bar
        self.stats_lbl.config(
            text=f"{len(self.graph.nodes)} nodes  ·  {len(self.graph.edges)} edges")

    def _draw_edge(self, edge):
        c = self.canvas
        a, b = edge.src, edge.dst
        hi = (edge is self.sel_edge)
        col = SEL_EDGE_COLOUR if hi else EDGE_COLOUR
        lw  = 2.5 if hi else 1.5

        if a is b:
            r = NODE_R + 14
            c.create_oval(a.x, a.y - r*1.9, a.x + r*1.7, a.y,
                          outline=col, width=lw)
            if edge.label:
                c.create_text(a.x + r*0.85, a.y - r*0.95,
                              text=edge.label, fill=EDGE_LABEL_COL,
                              font=("Helvetica",7,"bold"))
            return

        dx, dy = b.x - a.x, b.y - a.y
        dist = math.hypot(dx, dy) + 1e-6
        # shorten to node edge
        ex = b.x - dx/dist * (NODE_R + 6)
        ey = b.y - dy/dist * (NODE_R + 6)
        sx = a.x + dx/dist * (NODE_R + 2)
        sy = a.y + dy/dist * (NODE_R + 2)

        # perpendicular offset for curve
        perp = 22
        mx = (sx+ex)/2 + (-dy/dist)*perp
        my = (sy+ey)/2 + ( dx/dist)*perp

        c.create_line(sx, sy, mx, my, ex, ey,
                      fill=col, width=lw, smooth=True,
                      arrow=tk.LAST, arrowshape=(12,15,5))

        if edge.label:
            lx = (sx + mx + ex) / 3
            ly = (sy + my + ey) / 3
            # pill background
            c.create_text(lx, ly, text=edge.label,
                          fill=EDGE_LABEL_COL, font=("Helvetica",7,"bold"))

    def _draw_node(self, node):
        c = self.canvas
        x, y = node.x, node.y
        r = NODE_R
        col = node.colour
        sel = (node is self.sel_node) or (node is self.edge_src)

        # drop shadow
        c.create_oval(x-r+3,y-r+4,x+r+3,y+r+4,
                      fill="#08080f", outline="")

        # glow ring if selected
        if sel:
            for gr, gc in [(r+14, _lighten(col,0.05)),
                           (r+8,  _lighten(col,0.2))]:
                c.create_oval(x-gr,y-gr,x+gr,y+gr, fill=gc, outline="")

        # body gradient effect: outer ring slightly darker
        c.create_oval(x-r,y-r,x+r,y+r,
                      fill=_darken(col,0.18), outline="")
        inner = int(r * 0.82)
        c.create_oval(x-inner,y-inner,x+inner,y+inner,
                      fill=col, outline="")

        # border
        border = "#ffffff" if sel else _lighten(col, 0.5)
        bw     = 3 if sel else 1.5
        c.create_oval(x-r,y-r,x+r,y+r,
                      fill="", outline=border, width=bw)

        # label — two lines: name + (id)
        name = node.label
        nid  = f"({node.id})"
        c.create_text(x, y-6,  text=name, fill="#fff",
                      font=("Helvetica",8,"bold"), width=r*2-6)
        c.create_text(x, y+9,  text=nid,  fill=_lighten(col,0.7),
                      font=("Helvetica",7))

    # ── Canvas events ─────────────────────────────────────────────────────────
    def _node_at(self, x, y):
        for n in reversed(self.graph.nodes):
            if math.hypot(n.x-x, n.y-y) <= NODE_R+4: return n
        return None

    def _edge_near(self, x, y, tol=9):
        for e in self.graph.edges:
            a, b = e.src, e.dst
            if a is b: continue
            dx, dy = b.x-a.x, b.y-a.y
            t = ((x-a.x)*dx+(y-a.y)*dy)/(dx*dx+dy*dy+1e-9)
            t = max(0, min(1, t))
            if math.hypot(a.x+t*dx-x, a.y+t*dy-y) < tol: return e
        return None

    def _on_press(self, ev):
        x, y = ev.x, ev.y
        n = self._node_at(x, y)

        if self.edge_mode:
            if n:
                if self.edge_src is None:
                    self.edge_src = n
                    self.mode_lbl.config(text=f"⟶  now click target  (src: {n.label})")
                else:
                    self._open_edge_dialog(self.edge_src, n)
                    self._exit_edge_mode()
            return

        if n:
            self.sel_node = n; self.sel_edge = None
            self.drag_node = n
            self.drag_ox = x - n.x; self.drag_oy = y - n.y
            n.pinned = True
        else:
            e = self._edge_near(x, y)
            self.sel_edge = e; self.sel_node = None
        self._refresh_lists()

    def _on_drag(self, ev):
        self._mx, self._my = ev.x, ev.y
        if self.drag_node:
            self.drag_node.x = ev.x - self.drag_ox
            self.drag_node.y = ev.y - self.drag_oy

    def _on_release(self, ev):
        if self.drag_node:
            self.drag_node.pinned = False
            self.drag_node.vx = self.drag_node.vy = 0
            self.drag_node = None

    def _on_dblclick(self, ev):
        n = self._node_at(ev.x, ev.y)
        if n: self._dlg_rename_node(n)

    def _on_rclick(self, ev):
        n = self._node_at(ev.x, ev.y)
        e = None if n else self._edge_near(ev.x, ev.y)
        menu = tk.Menu(self.root, tearoff=0,
                       bg=CARD, fg=TEXT, activebackground=ACCENT,
                       activeforeground="#fff", relief=tk.FLAT, bd=0)
        if n:
            menu.add_command(label=f"Rename \"{n.label}\"",
                             command=lambda: self._dlg_rename_node(n))
            menu.add_command(label="Draw edge from here",
                             command=lambda: self._start_edge_from(n))
            menu.add_separator()
            menu.add_command(label=f"Delete \"{n.label}\"",
                             command=lambda: self._delete_node(n))
        elif e:
            menu.add_command(label=f"Edit label \"{e.label}\"",
                             command=lambda: self._dlg_rename_edge(e))
            menu.add_separator()
            menu.add_command(label="Delete edge",
                             command=lambda: self._delete_edge(e))
        else:
            menu.add_command(label="Add node here",
                             command=lambda: self._dlg_add_node(ev.x, ev.y))
        menu.post(ev.x_root, ev.y_root)

    # ── Dialogs ───────────────────────────────────────────────────────────────
    def _dlg_add_node(self, x=None, y=None):
        label = simpledialog.askstring("Add Node","Object label  (e.g. chair):",
                                       parent=self.root)
        if not label: return
        nx_ = x or random.uniform(self.canvas_w*.35, self.canvas_w*.65)
        ny_ = y or random.uniform(self.canvas_h*.35, self.canvas_h*.65)
        self.graph.add_node(label.strip(), x=nx_, y=ny_)
        self._refresh_lists()

    def _dlg_rename_node(self, n):
        v = simpledialog.askstring("Rename", f"New label for '{n.label}':",
                                   initialvalue=n.label, parent=self.root)
        if v: n.label=v.strip(); n.colour=node_col(n.label); self._refresh_lists()

    def _toggle_edge_mode(self):
        if self.edge_mode: self._exit_edge_mode()
        else:
            self.edge_mode = True
            self.edge_src  = None
            self._btns["add_edge"][1]("#f97316")   # highlight button orange
            self.mode_lbl.config(text="⟶  EDGE MODE:  click source node")

    def _exit_edge_mode(self):
        self.edge_mode = False; self.edge_src = None
        self._btns["add_edge"][1]("#2d2d44")       # restore button colour
        self.mode_lbl.config(text="")

    def _start_edge_from(self, n):
        self.edge_mode = True; self.edge_src = n
        self._btns["add_edge"][1]("#f97316")
        self.mode_lbl.config(text=f"⟶  now click target  (src: {n.label})")

    def _open_edge_dialog(self, src, dst):
        win = tk.Toplevel(self.root); win.title("Add Edge")
        win.configure(bg=CARD); win.grab_set()
        win.resizable(False, False)

        tk.Label(win, text=f"{src.label} ({src.id})  →  {dst.label} ({dst.id})",
                 bg=CARD, fg=TEXT, font=("Helvetica",11,"bold"),
                 padx=20, pady=14).pack()

        inner = tk.Frame(win, bg=CARD, padx=20); inner.pack(fill=tk.X)
        tk.Label(inner, text="Relationship:", bg=CARD, fg=SUBTEXT,
                 font=("Helvetica",9)).pack(anchor=tk.W)

        var = tk.StringVar()
        cb  = ttk.Combobox(inner, textvariable=var, width=26,
                           values=sorted(SPATIAL_RELS) + [
                               "same super category as","smaller than",
                               "bigger than","lower than","higher than",
                               "same material as","same style as",
                           ])
        cb.pack(pady=6); cb.focus_set()

        def ok():
            self.graph.add_edge(src, dst, var.get().strip())
            self._refresh_lists(); win.destroy()

        tk.Button(win, text="Add Edge", command=ok,
                  bg=ACCENT, fg="#fff", relief=tk.FLAT,
                  padx=16, pady=8, font=("Helvetica",10,"bold"),
                  cursor="hand2").pack(pady=12)
        win.bind("<Return>", lambda _: ok())

    def _dlg_rename_edge(self, e):
        v = simpledialog.askstring("Edit Label",
            f"Label for  {e.src.label} → {e.dst.label}:",
            initialvalue=e.label, parent=self.root)
        if v is not None: e.label=v.strip(); self._refresh_lists()

    def _del_sel_node(self):
        if self.sel_node: self._delete_node(self.sel_node)

    def _delete_node(self, n):
        if messagebox.askyesno("Delete",f"Remove '{n.label} ({n.id})' and its edges?",
                               parent=self.root):
            self.graph.remove_node(n)
            if self.sel_node is n: self.sel_node = None
            self._refresh_lists()

    def _del_sel_edge(self):
        if self.sel_edge: self._delete_edge(self.sel_edge)

    def _delete_edge(self, e):
        self.graph.remove_edge(e)
        if self.sel_edge is e: self.sel_edge = None
        self._refresh_lists()

    def _export(self):
        path = filedialog.asksaveasfilename(
            parent=self.root,
            initialdir=os.path.dirname(self.out_png),
            initialfile=os.path.basename(self.out_png),
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("All","*.*")],
            title="Export PNG")
        if path:
            export_png(self.graph, path, f"Scene Graph: {self.scene_id}")
            messagebox.showinfo("Exported", f"Saved:\n{path}", parent=self.root)

    def _load_json(self):
        path = filedialog.askopenfilename(parent=self.root,
            title="Load relationships JSON",
            initialdir=os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "data"),
            filetypes=[("JSON","*.json"),("All","*.*")])
        if not path: return
        sid = simpledialog.askstring("Scene ID","Scan ID to load:",
                                     parent=self.root)
        if not sid: return
        g = load_scene(path, sid.strip())
        if g is None:
            messagebox.showerror("Not found",
                f"'{sid}' not found in file.", parent=self.root); return
        self.graph = g; self.scene_id = sid
        self.sel_node = self.sel_edge = None
        self._centered = False          # re-centre on next tick
        self.root.title(f"Scene Graph Editor  ·  {sid}")
        self._refresh_lists()

    def _show_json(self):
        text = json.dumps(self.graph.to_dict(), indent=2)
        win = tk.Toplevel(self.root); win.title("Scene Graph JSON")
        win.configure(bg=CARD)
        fr = tk.Frame(win, bg=CARD, padx=8, pady=8); fr.pack(fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(fr); sb.pack(side=tk.RIGHT, fill=tk.Y)
        tx = tk.Text(fr, bg="#0d0d1a", fg="#a6e3a1", font=("Courier",9),
                     yscrollcommand=sb.set, wrap=tk.NONE, relief=tk.FLAT)
        tx.insert(tk.END, text); tx.config(state=tk.DISABLED)
        tx.pack(fill=tk.BOTH, expand=True); sb.config(command=tx.yview)
        bf = tk.Frame(win, bg=CARD, pady=6); bf.pack()
        def copy():
            win.clipboard_clear(); win.clipboard_append(text)
        tk.Button(bf, text="Copy", command=copy, bg=ACCENT, fg="#fff",
                  relief=tk.FLAT, padx=12, pady=4, cursor="hand2").pack(side=tk.LEFT,padx=6)
        tk.Button(bf, text="Close", command=win.destroy, bg=BORDER, fg=TEXT,
                  relief=tk.FLAT, padx=12, pady=4, cursor="hand2").pack(side=tk.LEFT)

    # ── Sidebar refresh ───────────────────────────────────────────────────────
    def _refresh_lists(self):
        self.node_lb.delete(0, tk.END)
        for n in self.graph.nodes:
            self.node_lb.insert(tk.END, f"  {n.label}  ({n.id})")
            if n is self.sel_node:
                self.node_lb.itemconfig(tk.END, fg=ACCENT)

        filt = self._filter_var.get().lower().strip()
        if filt == "filter edges…": filt = ""

        vis_edges = [e for e in self.graph.edges
                     if not filt or filt in e.label.lower()
                        or filt in e.src.label.lower()
                        or filt in e.dst.label.lower()]

        self.edge_lb.delete(0, tk.END)
        for e in vis_edges:
            lbl = f"  {e.src.label}({e.src.id}) → {e.dst.label}({e.dst.id})"
            if e.label: lbl += f"  [{e.label}]"
            self.edge_lb.insert(tk.END, lbl)
            if e is self.sel_edge:
                self.edge_lb.itemconfig(tk.END, fg=SEL_EDGE_COLOUR)

        self.edge_count_lbl.config(
            text=f"{len(vis_edges)}/{len(self.graph.edges)}"
        )

    def _on_node_lb(self, ev):
        sel = self.node_lb.curselection()
        if sel and sel[0] < len(self.graph.nodes):
            self.sel_node = self.graph.nodes[sel[0]]
            self.sel_edge = None

    def _on_edge_lb(self, ev):
        sel = self.edge_lb.curselection()
        if sel and sel[0] < len(self.graph.edges):
            self.sel_edge = self.graph.edges[sel[0]]
            self.sel_node = None

    def on_close(self):
        self.running = False; self.root.destroy()


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    _HERE       = os.path.dirname(os.path.abspath(__file__))
    _DATA_DIR   = os.path.join(_HERE, "data")
    _OUTPUT_DIR = os.path.join(_HERE, "output")
    os.makedirs(_DATA_DIR,   exist_ok=True)
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(description="Scene Graph Editor")
    parser.add_argument("--scene",  default="DiningRoom-31158")
    parser.add_argument("--rel",    default=os.path.join(
                                        _DATA_DIR,"relationships_diningroom_test.json"))
    parser.add_argument("--out",    default=os.path.join(
                                        _OUTPUT_DIR,"DiningRoom-31158_scene_graph.png"))
    parser.add_argument("--all-edges", action="store_true",
                        help="Load all edge types (default: spatial only)")
    args = parser.parse_args()

    graph = load_scene(args.rel, args.scene, spatial_only=not args.all_edges)
    if graph is None:
        print(f"[warn] '{args.scene}' not found in '{args.rel}' — using demo graph.")
        graph = demo_graph(); scene_id = "Demo"
    else:
        scene_id = args.scene
        print(f"[info] Loaded {len(graph.nodes)} nodes, "
              f"{len(graph.edges)} edges  ({scene_id})")

    root = tk.Tk()
    root.geometry("1280x800")

    app = App(root, graph, scene_id, args.out)
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    if HAS_MPL:
        def _bg():
            time.sleep(2.0)
            export_png(graph, args.out, f"Scene Graph: {scene_id}")
        threading.Thread(target=_bg, daemon=True).start()

    root.mainloop()


if __name__ == "__main__":
    main()
