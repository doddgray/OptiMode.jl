# using Plots: plot, plot!, heatmap, heatmap!, @layout, cgrad
# using Plots.PlotMeasures
# # using Plots.grid
# using Plots: plot, plot!, heatmap, @layout, cgrad, grid, heatmap!
# using AbstractPlotting, # GLMakie
# using UnicodePlots

export plot_ε, compare_fields, plot_field, plot_d⃗, ylims

################################################################################
#                              Utility Functions                               #
################################################################################








# ε plotting functions

function plot_ε(ε1,x::AbstractVector{Float64},y::AbstractVector{Float64}; outlines=false,cmap_diag=cgrad(:viridis),cmap_offdiag=cgrad(:bluesreds),scale_each_component=true)
	tmi,bmi,lmi,rmi = -4mm, -4mm, -4mm, -4mm
	tmo,bmo,lmo,rmo = 2mm, 0mm, 0mm, 0mm

	labels = [	"ε₁₁"	"ε₁₂"	"ε₁₃"
				"ε₂₁"	"ε₂₂"	"ε₂₃"
				"ε₃₁"	"ε₃₂"	"ε₃₃"	]

	showaxes = [
			:y 	:no :no
			:y 	:no :no
		 	:yes :x :x
	]
	lm = [
			lmo lmi lmi
			lmo lmi lmi
			lmo lmi lmi
	]
	rm = [
			rmi rmi rmo
			rmi rmi rmo
			rmi rmi rmo
	]
	tm = [
			tmo tmo tmo
			tmi tmi tmi
			tmi tmi tmi
	]
	bm = [
			bmi bmi bmi
			bmi bmi bmi
			bmo bmo bmo
	]


	plot_grid =[
		heatmap(
			x,
			y,
			# transpose([real(ε1[a,b,1][i,j]) for a=1:length(x),b=1:length(y)]),
			transpose([real(ε1[i,j,a,b,1]) for a=1:length(x),b=1:length(y)]),
			c = (i==j ? cmap_diag : cmap_offdiag), #cgrad(:cherry),
			aspect_ratio=:equal,
			colorbar = false,
			showaxis = showaxes[j,i],
			top_margin= tm[j,i],
			bottom_margin=bm[j,i],
			left_margin = lm[j,i],
			right_margin = rm[j,i],
			# clim = ( min(ε1...), max(ε1[:,:,i,j]) ),
			clim = ( i==j ? (scale_each_component ? ( 0., maximum(real(ε1[i,j,:,:,1]))) : ( 0., maximum(real(ε1))) ) : (scale_each_component ? ( min( minimum(real(ε1[i,j,:,:,1])), -maximum(real(ε1[i,j,:,:,1]))), max( maximum(real(ε1[i,j,:,:,1])), -minimum(real(ε1[i,j,:,:,1])) ) ) : ( minimum(real(ε1)), maximum(real(ε1))))),
		) for i=1:3,j=1:3 ]

	zoom_plot = heatmap(
			x,
			y,
			# transpose([real(ε1[a,b,1][1,1]) for a=1:length(x),b=1:length(y)]),
			transpose([real(ε1[1,1,a,b,1]) for a=1:length(x),b=1:length(y)]),
			# transpose(ε1[:,:][1,1]),
			c=cmap_diag, #cgrad(:cherry),
			aspect_ratio=:equal,
			legend=false,
			colorbar = true,
			clim = (scale_each_component ? ( 0., maximum(real(ε1[1,1,:,:,1]))) : ( 0., maximum(real(ε1)))),
		)

	if outlines
		plot!(zoom_plot,shape(b),color=nothing,linecolor=:yellow)
		plot!(zoom_plot,shape(s),color=nothing,linecolor=:black)
		plot!(zoom_plot,shape(t),color=nothing,linecolor=:black)
	end
	l = @layout [  a
					grid(3,3)
				]
	plot(
		zoom_plot,
		plot_grid...,
		layout=l,
		size = (600,850),
	)
end

function plot_ε(ε1,g::MaxwellGrid; outlines=false, cmap=cgrad(:cherry))
	plot_ε(ε1,g.x,g.y,outlines=outlines,cmap=cmap)
end

function plot_ε(shapes::AbstractVector{GeometryPrimitives.Shape{2,4,D}},Δx=6.,Δy=4.,Nx=64,Ny=64; outlines=false, cmap=cgrad(:cherry)) where D
	g = MaxwellGrid(Δx,Δy,Nx,Ny)
	plot_ε(εₛ(shapes,g),g.x,g.y,outlines=outlines,cmap=cmap)
end

function plot_ε(shapes::AbstractVector{GeometryPrimitives.Shape{2,4,D}},g::MaxwellGrid; outlines=false, cmap=cgrad(:cherry)) where D
	plot_ε(εₛ(shapes,g),g.x,g.y,outlines=outlines,cmap=cmap)
end


################################################################################
#                        Modesolver plotting functions                         #
################################################################################



function compare_fields(f_mpb,f,ds)
	xlim = (minimum(ds.x),maximum(ds.x))
	ylim = (minimum(ds.y),maximum(ds.y))

    hm_f_mpb_real = [ heatmap(ds.x,ds.y,[real(f_mpb[i,j][ix]) for i=1:ds.Nx,j=1:ds.Ny]',aspect_ratio=:equal,c=cgrad(:RdBu),xlim=xlim,ylim=ylim) for ix=1:3]
    hm_f_mpb_imag = [ heatmap(ds.x,ds.y,[imag(f_mpb[i,j][ix]) for i=1:ds.Nx,j=1:ds.Ny]',aspect_ratio=:equal,c=cgrad(:viridis),xlim=xlim,ylim=ylim) for ix=1:3]

    hm_f_real = [ heatmap(ds.x,ds.y,[real(f[ix,i,j]) for i=1:ds.Nx,j=1:ds.Ny]',aspect_ratio=:equal,c=cgrad(:RdBu),xlim=xlim,ylim=ylim) for ix=1:3]
    hm_f_imag = [ heatmap(ds.x,ds.y,[imag(f[ix,i,j]) for i=1:ds.Nx,j=1:ds.Ny]',aspect_ratio=:equal,c=cgrad(:viridis),xlim=xlim,ylim=ylim) for ix=1:3]

    hm_f_ratio_real = [ heatmap(ds.x,ds.y,[real(f[ix,i,j])/real(f_mpb[i,j][ix]) for i=1:ds.Nx,j=1:ds.Ny]',aspect_ratio=:equal,c=cgrad(:RdBu),xlim=xlim,ylim=ylim) for ix=1:3]
    hm_f_ratio_imag = [ heatmap(ds.x,ds.y,[imag(f[ix,i,j])/imag(f_mpb[i,j][ix]) for i=1:ds.Nx,j=1:ds.Ny]',aspect_ratio=:equal,c=cgrad(:viridis),xlim=xlim,ylim=ylim) for ix=1:3]

    l = @layout [   a   b   c
                    d   e   f
                    g   h   i
                    k   l   m
                    n   o   p
                    q   r   s    ]
    plot(hm_f_mpb_real...,
        hm_f_mpb_imag...,
        hm_f_real...,
        hm_f_imag...,
        hm_f_ratio_real...,
        hm_f_ratio_imag...,
        layout=l,
        size = (1300,1300),
    )
end

function plot_field(f::Array{ComplexF64,4},ds::MaxwellGrid;zind=1)
	xlim = (minimum(ds.x),maximum(ds.x))
	ylim = (minimum(ds.y),maximum(ds.y))

    hm_f_real = [ heatmap(ds.x,ds.y,[real(f[ix,i,j,zind]) for i=1:ds.Nx,j=1:ds.Ny]',aspect_ratio=:equal,c=cgrad(:RdBu),xlim=xlim,ylim=ylim) for ix=1:3]
    hm_f_imag = [ heatmap(ds.x,ds.y,[imag(f[ix,i,j,zind]) for i=1:ds.Nx,j=1:ds.Ny]',aspect_ratio=:equal,c=cgrad(:viridis),xlim=xlim,ylim=ylim) for ix=1:3]

    l = @layout [   a   b
                    c   d
					e   f    ]

	# l = @layout [   a   b   c
    #                 d   e	f 	]

    plot(
        hm_f_real[1],
        hm_f_imag[1],
		hm_f_real[2],
        hm_f_imag[2],
		hm_f_real[3],
        hm_f_imag[3],
        layout=l,
        size = (1000,800),
    )
end

function plot_d⃗(H::Array{ComplexF64,2},k::Float64,g::MaxwellGrid;zind=1)
	mn,kpg_mag = calc_kpg(k,g.Δx,g.Δy,g.Δz,g.Nx,g.Ny,g.Nz)
	d = fft(kx_t2c(reshape(H,(2,g.Nx,g.Ny,g.Nz)),kpg_mag,mn),(2:4))
	plot_field(d,g;zind)
end
