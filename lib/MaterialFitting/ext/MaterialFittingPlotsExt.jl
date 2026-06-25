module MaterialFittingPlotsExt

using MaterialFitting
using MaterialFitting: SellmeierFit, ThermoSellmeierFit, sellmeier_n, sellmeier_n², thermo_n²
using Printf
using Plots

function MaterialFitting.plot_sellmeier_fit(sf::SellmeierFit; path=nothing, show::Bool=false)
    ds = sf.dataset
    lo, hi = sf.λ_range
    # plot over the data span (which already extends a little beyond the validity range);
    # guard against the model going below a UV pole, where n² < 0, with NaN.
    λext = range(first(ds.λ), last(ds.λ); length=500)
    nfit = [(v = sellmeier_n²(sf, l); v > 0 ? sqrt(v) : NaN) for l in λext]

    m = lo .≤ ds.λ .≤ hi
    resid = [sellmeier_n(sf, l) for l in ds.λ[m]] .- ds.n[m]

    ttl = @sprintf("%s — %d-term Sellmeier  (RMS Δn=%.1e, max Δn=%.1e)",
                   isempty(sf.name) ? "fit" : sf.name, sf.n_terms, sf.rms_error, sf.max_error)
    p1 = plot(λext, nfit; lw=2, color=:dodgerblue, label="Sellmeier fit",
              ylabel="refractive index n", title=ttl, titlefontsize=9, legend=:best,
              framestyle=:box)
    vspan!(p1, [lo, hi]; color=:seagreen, alpha=0.12, label="validity range [$lo, $hi] μm")
    scatter!(p1, ds.λ, ds.n; ms=3, color=:black, markerstrokewidth=0,
             label=isempty(ds.label) ? "data" : "data: $(ds.label)")

    p2 = scatter(ds.λ[m], resid; ms=3, color=:crimson, markerstrokewidth=0, label="",
                 xlabel="vacuum wavelength λ (μm)", ylabel="Δn (fit − data)", framestyle=:box)
    hline!(p2, [0.0]; color=:gray, ls=:dash, label="")

    plt = plot(p1, p2; layout=grid(2, 1; heights=[0.72, 0.28]), size=(760, 620), link=:x)
    path !== nothing && savefig(plt, path)
    show && display(plt)
    return plt
end

function MaterialFitting.plot_thermo_fit(tf::ThermoSellmeierFit; path=nothing, show::Bool=false)
    lo, hi = tf.λ_range
    λext = range(lo, hi; length=400)
    ttl = @sprintf("%s — thermo Sellmeier (%d terms, T-order %d, RMS Δn=%.1e)",
                   isempty(tf.name) ? "fit" : tf.name, tf.n_terms, tf.T_poly_order, tf.rms_error)
    plt = plot(; xlabel="vacuum wavelength λ (μm)", ylabel="refractive index n",
               title=ttl, titlefontsize=9, legend=:best, framestyle=:box, size=(760, 520))
    vspan!(plt, [lo, hi]; color=:seagreen, alpha=0.08, label="")
    pal = palette(:thermal, length(tf.fits))
    for (k, f) in enumerate(tf.fits)
        T = tf.temperatures[k]
        plot!(plt, λext, [sqrt(max(thermo_n²(tf, l, T), 0.0)) for l in λext];
              lw=2, color=pal[k], label=@sprintf("model T=%.0f°C", T))
        ds = f.dataset
        scatter!(plt, ds.λ, ds.n; ms=2.5, color=pal[k], markerstrokewidth=0, label="")
    end
    path !== nothing && savefig(plt, path)
    show && display(plt)
    return plt
end

end # module
