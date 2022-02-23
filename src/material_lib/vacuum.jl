################################################################################
#                                   vacuum                                     #
################################################################################
export Vacuum

"""
This code creates a trivial Material model representing vacuum
"""

function make_Vacuum()
	@variables ω, λ, T
	models = Dict([
		:n		=>	1,
		:ng		=>	1,
		:gvd	=>	0,
		:ε 		=> 	diagm([1,1,1]),
	])
	defaults =	Dict([
		:ω		=>		inv(0.8),	# μm⁻¹
		:λ		=>		0.8,		# μm
		:T		=>		20.0,	    # °C

	])
	Material(models, defaults, :Vacuum, colorant"black")
end

################################################################

Vacuum = make_Vacuum()