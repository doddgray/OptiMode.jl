# using Plots: plot, plot!, heatmap, heatmap!, @layout, cgrad
using Plots.PlotMeasures
# using Plots.grid
using Plots

export plot_ε, compare_fields


# shape plotting functions 
function shape(b::Box)
    xc,yc = b.c
    r1,r2 = b.r
    A = inv(b.p) .* b.r'
    e1 = A[:,1] / √sum([a^2 for a in A[:,1]])
    e2 = A[:,2] / √sum([a^2 for a in A[:,2]])
    pts = [  r1*e1 + r2*e2,
             r1*e1 - r2*e2,
            -r1*e1 - r2*e2,
            -r1*e1 + r2*e2,
        ]
    b_shape = Plots.Shape([Tuple(pt) for pt in pts])
end	
shape(p::GeometryPrimitives.Polygon) = Plots.Shape([Tuple(p.v[i,:]) for i in range(1,length(p.v[:,1]),step=1)])
shape(s::GeometryPrimitives.Sphere) = Plots.partialcircle(0, 2π, 100, s.r)


# ε plotting functions

function plot_ε(ε1,x::AbstractVector{Float64},y::AbstractVector{Float64}; outlines=false, cmap=cgrad(:cherry))
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
			transpose([real(ε1[a,b][i,j]) for a=1:length(x),b=1:length(y)]),
			c=cmap, #cgrad(:cherry),
			aspect_ratio=:equal,
			colorbar = false,
			showaxis = showaxes[j,i],
			top_margin= tm[j,i],
			bottom_margin=bm[j,i],
			left_margin = lm[j,i],
			right_margin = rm[j,i],
			# clim = ( min(ε1...), max(ε1[:,:,i,j]) ),
			# clim = ( 0., max(ε1[:,:][i,j]...),
		) for i=1:3,j=1:3 ]
	
	zoom_plot = heatmap(
			x,
			y,
			transpose([real(ε1[a,b][1,1]) for a=1:length(x),b=1:length(y)]),
			# transpose(ε1[:,:][1,1]),
			c=cmap, #cgrad(:cherry),
			aspect_ratio=:equal,
			legend=false,
			colorbar = true,
			# clim = ( 0., max(ε1...) ),
		)
	
	if outlines
		plot!(zoom_plot,shape(b),color=nothing,linecolor=:yellow)
		plot!(zoom_plot,shape(s),color=nothing,linecolor=:black)
		plot!(zoom_plot,shape(t),color=nothing,linecolor=:black)
	end	
	l = @layout [  a 
					Plots.grid(3,3)
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
	g = Grid(Δx,Δy,Nx,Ny)
	plot_ε(εₛ(shapes,g),g.x,g.y,outlines=outlines,cmap=cmap)
end

function plot_ε(shapes::AbstractVector{GeometryPrimitives.Shape{2,4,D}},g::MaxwellGrid; outlines=false, cmap=cgrad(:cherry)) where D
	plot_ε(εₛ(shapes,g),g.x,g.y,outlines=outlines,cmap=cmap)
end

# Modesolver result plotting functions

function compare_fields(f_mpb,f,xlim,ylim)
    hm_f_mpb_real = [ heatmap(x_mpb,y_mpb,[real(f_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:RdBu),xlim=xlim,ylim=ylim) for ix=1:3]
    hm_f_mpb_imag = [ heatmap(x_mpb,y_mpb,[imag(f_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:viridis),xlim=xlim,ylim=ylim) for ix=1:3]
    
    hm_f_real = [ heatmap(x_mpb,y_mpb,[real(f[ix,i,j]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:RdBu),xlim=xlim,ylim=ylim) for ix=1:3]
    hm_f_imag = [ heatmap(x_mpb,y_mpb,[imag(f[ix,i,j]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:viridis),xlim=xlim,ylim=ylim) for ix=1:3]
    
    hm_f_ratio_real = [ heatmap(x_mpb,y_mpb,[real(f[ix,i,j])/real(f_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:RdBu),xlim=xlim,ylim=ylim) for ix=1:3]
    hm_f_ratio_imag = [ heatmap(x_mpb,y_mpb,[imag(f[ix,i,j])/imag(f_mpb[i,j][ix]) for i=1:nx_mpb,j=1:ny_mpb]',aspect_ratio=:equal,c=cgrad(:viridis),xlim=xlim,ylim=ylim) for ix=1:3]
    
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