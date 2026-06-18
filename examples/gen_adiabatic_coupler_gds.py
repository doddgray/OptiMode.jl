"""Generate an adiabatic directional coupler layout with GDSFactory and export GDS.

Two single-mode strip waveguides start far apart and adiabatically approach each
other, so the even/odd supermodes evolve slowly along the device and power
transfers between the guides. The layout is written to ``adiabatic_coupler.gds``
(GDS layer ``(1, 0)`` = the waveguide core), which the Julia EME example/test
imports with ``EigenmodeExpansion.read_gds``.

Run:  python examples/gen_adiabatic_coupler_gds.py [output.gds]
"""

import sys

import gdsfactory as gf


def adiabatic_coupler(
    length: float = 20.0,
    width: float = 0.5,
    gap0: float = 1.2,
    gap1: float = 0.3,
    layer=(1, 0),
) -> gf.Component:
    """Two strips whose centre-to-centre gap narrows linearly from gap0 to gap1.

    Each strip is a parallelogram (constant width, linearly shifting centre line)
    so the cross-section at any propagation position is two rectangles — exactly
    what the EME importer slices into cells.
    """
    c = gf.Component("adiabatic_coupler")
    L = length

    def strip(sign: int):
        # centre line of this strip at the two ends
        y0 = sign * (gap0 / 2 + width / 2)
        y1 = sign * (gap1 / 2 + width / 2)
        # vertices CCW: top edge then bottom edge
        return [
            (0.0, y0 + width / 2),
            (L, y1 + width / 2),
            (L, y1 - width / 2),
            (0.0, y0 - width / 2),
        ]

    c.add_polygon(strip(+1), layer=layer)
    c.add_polygon(strip(-1), layer=layer)
    return c


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "adiabatic_coupler.gds"
    c = adiabatic_coupler()
    # flatten so all geometry is present as boundaries (no references) for the
    # minimal GDS reader on the Julia side
    c = c.flatten()
    c.write_gds(out)
    print(f"wrote {out}  ({len(c.get_polygons())} polygon layer groups)")


if __name__ == "__main__":
    main()
