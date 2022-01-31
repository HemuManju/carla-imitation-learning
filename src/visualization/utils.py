def change_spine_asthetics(axes):
    for spine in ['right', 'top']:
        for ax in axes:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(0.5)
            ax.spines[spine].set_color('#BABABA')
    return None
