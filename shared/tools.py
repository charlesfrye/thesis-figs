def add_panel_label(label, ax, pos=(-0.1, 1.15), size="x-large"):
    ax.text(*pos, label, transform=ax.transAxes,
            size=size,
            fontweight='bold', va='top', ha='right')