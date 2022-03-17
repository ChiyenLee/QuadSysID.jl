using ApproxFunc

times = 0:0.1:10
values = cos.(times*1)

S=Fourier(Interval(0,2*pi))
x = 0:0.1:2*pi
v= 1.0*cos.(x ) #.+ 3.0*sin.(1.0*x)
f=Fun(S,ApproxFun.transform(S,v))
plot(x,v)
plot(x, f.(x))

# Taking derivative 
D = Derivative(S)
df = D*f
plot(x, df.(x))

# x = points(S,100)
x = 0:0.005:2*pi
v= 1.0*cos.(x .+ 0.43232) 
f=Fun(S,ApproxFun.transform(S,v))
