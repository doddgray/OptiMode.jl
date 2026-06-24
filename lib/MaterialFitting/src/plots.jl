export plot_sellmeier_fit, plot_thermo_fit

# Plotting is provided by the package extension `MaterialFittingPlotsExt`, loaded when
# `Plots` is available. These are the generic stubs whose methods that extension defines;
# the fitting routines call them through the `_emit_*` helpers below so that a fit-vs-data
# comparison plot is saved every time a fit is run (whenever `Plots` is loaded).

"""
    plot_sellmeier_fit(fit::SellmeierFit; path=nothing, show=false)

Plot a fitted Sellmeier model against its input data: the index `n(λ)` over (and beyond)
the validity range with the fit range shaded, plus a residual panel. Saves a PNG to `path`
if given and returns the plot object. **Requires `Plots` to be loaded** (provided by the
`MaterialFittingPlotsExt` extension).
"""
function plot_sellmeier_fit end

"""
    plot_thermo_fit(tf::ThermoSellmeierFit; path=nothing, show=false)

Plot a temperature-dependent Sellmeier model against its multi-temperature input data
(one curve per temperature) with the validity range shaded. Requires `Plots`.
"""
function plot_thermo_fit end

_plots_loaded() = !isempty(methods(plot_sellmeier_fit))

function _plot_filename(name, sf::SellmeierFit)
    base = name !== nothing ? string(name) :
           (isempty(sf.name) ? "sellmeier_fit" : replace(sf.name, r"[^\w\-]+" => "_"))
    ax = isempty(sf.axis) ? "" : "_$(sf.axis)"
    return "$(base)$(ax)_$(sf.n_terms)term.png"
end

function _emit_fit_plot(sf::SellmeierFit; dir=nothing, name=nothing, show=false)
    if !_plots_loaded()
        @warn "MaterialFitting: load `Plots` (`using Plots`) to auto-generate fit-comparison plots; none saved." maxlog = 1
        return nothing
    end
    path = nothing
    if dir !== nothing
        isdir(dir) || mkpath(dir)
        path = joinpath(dir, _plot_filename(name, sf))
    end
    return plot_sellmeier_fit(sf; path=path, show=show)
end

function _emit_thermo_plot(tf::ThermoSellmeierFit; dir=nothing, show=false)
    _plots_loaded() || return nothing
    path = nothing
    if dir !== nothing
        isdir(dir) || mkpath(dir)
        base = isempty(tf.name) ? "thermo_sellmeier_fit" : replace(tf.name, r"[^\w\-]+" => "_")
        path = joinpath(dir, "$(base)_thermo.png")
    end
    return plot_thermo_fit(tf; path=path, show=show)
end
