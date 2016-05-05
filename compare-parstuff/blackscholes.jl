#This example has been adopted from https://github.com/IntelLabs/ParallelAccelerator.jl/blob/master/examples/black-scholes/black-scholes.jl

using ArrayFire 
using Base.Threads
using ParallelAccelerator
using DistributedArrays

function blackscholes_vec(sptprice::AbstractArray{Float32},
                           strike::AbstractArray{Float32},
                           rate::AbstractArray{Float32},
                           volatility::AbstractArray{Float32},
                           time::AbstractArray{Float32})
    logterm = log10(sptprice ./ strike)
    powterm = .5 .* volatility .* volatility
    den = volatility .* sqrt(time)
    d1 = (((rate .+ powterm) .* time) .+ logterm) ./ den
    d2 = d1 .- den
    NofXd1 = cndf2(d1)
    NofXd2 = cndf2(d2)
    futureValue = strike .* exp(- rate .* time)
    c1 = futureValue .* NofXd2
    call = sptprice .* NofXd1 .- c1
    put  = call .- futureValue .+ sptprice
end

function blackscholes_devec(sptprice::Vector{Float32}, strike::Vector{Float32}, rate::Vector{Float32}, volatility::Vector{Float32}, time::Vector{Float32})
    sqt = sqrt(time)
    put = similar(strike)
    for i = 1:size(sptprice, 1)
        logterm = log10(sptprice[i] / strike[i])
        powterm = 0.5 * volatility[i] * volatility[i]
        den = volatility[i] * sqt[i]
        d1 = (((rate[i] + powterm) * time[i]) + logterm) / den
        d2 = d1 - den
        NofXd1 = 0.5 + 0.5 * erf(0.707106781 * d1)
        NofXd2 = 0.5 + 0.5 * erf(0.707106781 * d2)
        futureValue = strike[i] * exp(-rate[i] * time[i])
        c1 = futureValue * NofXd2
        call = sptprice[i] * NofXd1 - c1
        put[i] = call - futureValue + sptprice[i]
    end
    put
end

function blackscholes_distributed(sptprice::DArray{Float32}, strike::DArray{Float32}, rate::DArray{Float32}, volatility::DArray{Float32}, time::DArray{Float32})
    logterm = log10(sptprice ./ strike)
    powterm = .5 .* volatility .* volatility
    den = volatility .* sqrt(time)
    d1 = (((rate .+ powterm) .* time) .+ logterm) ./ den
    d2 = d1 .- den
    NofXd1 = cndf2(d1)
    NofXd2 = cndf2(d2)
    futureValue = strike .* exp((0-rate) .* time)
    c1 = futureValue .* NofXd2
    call = sptprice .* NofXd1 .- c1
    put  = call .- futureValue .+ sptprice
end

function blackscholes_mt(sptprice::Vector{Float32}, strike::Vector{Float32}, rate::Vector{Float32}, volatility::Vector{Float32}, time::Vector{Float32})
    sqt = sqrt(time)
    put = similar(strike)
    @threads for i = 1:size(sptprice, 1)
        logterm = log10(sptprice[i] / strike[i])
        powterm = 0.5 * volatility[i] * volatility[i]
        den = volatility[i] * sqt[i]
        d1 = (((rate[i] + powterm) * time[i]) + logterm) / den
        d2 = d1 - den
        NofXd1 = 0.5 + 0.5 * erf(0.707106781 * d1)
        NofXd2 = 0.5 + 0.5 * erf(0.707106781 * d2)
        futureValue = strike[i] * exp(-rate[i] * time[i])
        c1 = futureValue * NofXd2
        call = sptprice[i] * NofXd1 - c1
        put[i] = call - futureValue + sptprice[i]
    end
    put
end

@inline function cndf2(in::AbstractArray{Float32})
    out = 0.5 .+ 0.5 .* erf(0.707106781 .* in)
    return out
end


@acc begin 

@inline function cndf2_acc(in::AbstractArray{Float32})
    out = 0.5 .+ 0.5 .* erf(0.707106781 .* in)
    return out
end

function blackscholes_acc(sptprice::AbstractArray{Float32},
                           strike::AbstractArray{Float32},
                           rate::AbstractArray{Float32},
                           volatility::AbstractArray{Float32},
                           time::AbstractArray{Float32})
    logterm = log10(sptprice ./ strike)
    powterm = .5 .* volatility .* volatility
    den = volatility .* sqrt(time)
    d1 = (((rate .+ powterm) .* time) .+ logterm) ./ den
    d2 = d1 .- den
    NofXd1 = cndf2_acc(d1)
    NofXd2 = cndf2_acc(d2)
    futureValue = strike .* exp(- rate .* time)
    c1 = futureValue .* NofXd2
    call = sptprice .* NofXd1 .- c1
    put  = call .- futureValue .+ sptprice
end


end

function main(iterations)
    sptprice   = Float32[ 42.0 for i = 1:iterations ]
    initstrike = Float32[ 40.0 + (i / iterations) for i = 1:iterations ]
    rate       = Float32[ 0.5 for i = 1:iterations ]
    volatility = Float32[ 0.2 for i = 1:iterations ]
    time       = Float32[ 0.5 for i = 1:iterations ]
    tic()
    put1 = blackscholes_vec(sptprice, initstrike, rate, volatility, time)
    t1 = toq()
    tic()
    put2 = blackscholes_devec(sptprice, initstrike, rate, volatility, time)
    t2 = toq()
    tic()
    sptprice1 = distribute(sptprice)
    initstrike1 = distribute(initstrike)
    rate1 = distribute(rate)
    volatility1 = distribute(volatility)
    time1 = distribute(time)
    t31 = toq()
    tic()
    put3 = blackscholes_distributed(sptprice1, initstrike1, rate1, volatility1, time1)
    t32 = toq()
    tic()
    Array(put3)
    t33 = toq()
    tic()
    sptprice1 = AFArray(sptprice)
    initstrike1 = AFArray(initstrike)
    rate1 = AFArray(rate)
    volatility1 = AFArray(volatility)
    time1 = AFArray(time)
    t41 = toq()
    tic()
    put4 = blackscholes_vec(sptprice1, initstrike1, rate1, volatility1, time1)
    t42 = toq()
    tic()
    Array(put4)
    t43 = toq()
    tic()
    put5 = blackscholes_mt(sptprice, initstrike, rate, volatility, time)
    t5 = toq() 
    tic()
    put6 = blackscholes_acc(sptprice, initstrike, rate, volatility, time)
    t6 = toq() 
    println("Vectorized checksum: ", sum(put1))
    println("Devectorized checksum: ", sum(put2))
    println("Distributed checksum: ", sum(put3))
    println("GPU checksum: ", sum(put4))
    println("Multithreaded checksum: ", sum(put5))
    println("Paracc checksum: ", sum(put6))
    return t1,t2,t31,t32,t33,t41,t42,t43,t5,t6
end

function driver()
    srand(0)
    tic()
    iterations = 10^7
    blackscholes_vec(Float32[], Float32[], Float32[], Float32[], Float32[])
    blackscholes_vec(AFArray(Float32[1]), AFArray(Float32[1]), AFArray(Float32[1]), AFArray(Float32[1]), AFArray(Float32[1]))
    blackscholes_devec(Float32[], Float32[], Float32[], Float32[], Float32[])
    blackscholes_mt(Float32[], Float32[], Float32[], Float32[], Float32[])
    blackscholes_acc(Float32[], Float32[], Float32[], Float32[], Float32[])
    blackscholes_distributed(distribute(Float32[1]), distribute(Float32[1]), distribute(Float32[1]), distribute(Float32[1]), distribute(Float32[1]))
    println("SELFPRIMED ", toq())
    tvec, tdevec, td1, td2, td3, tg1, tg2, tg3, tmt, tacc = main(iterations)
    println("Time taken for vectorized version = $tvec")
    println("Time taken for devectorized version = $tdevec")
    println("Distributed Version:")
    println("Time taken for distribute step = $td1")
    println("Time taken for distibuted compute = $td2")
    println("Time taken for localize step = $td3")
    println("Heterogeneous Version: ")
    println("Time taken for CPU to GPU transfer = $tg1")
    println("Time taken for GPU compute = $tg2")
    println("Time taken for GPU to CPU transfer = $tg3")
    println("Time taken for multithreaded version= $tmt")
    println("Time taken for parallel acc version= $tacc")
    println("Vectorized rate = ", iterations / tvec, " opts/sec")
    println("Devectorized rate = ", iterations / tdevec, " opts/sec")
    println("Distributed rate = ", iterations / td2, " opts/sec")
    println("GPU rate = ", iterations / tg2, " opts/sec")
    println("Multithreaded rate = ", iterations / tmt, " opts/sec")
    println("Par acc rate = ", iterations / tacc, " opts/sec")
end
driver()
