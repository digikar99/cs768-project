
using LinearAlgebra

function argmaximum(f::Function, it)
    max_so_far = typemin(Float64)
    arg        = nothing
    for i in it
        score = f(i)
        if score > max_so_far
            max_so_far = score
            arg        = i
        end
    end
    arg
end


