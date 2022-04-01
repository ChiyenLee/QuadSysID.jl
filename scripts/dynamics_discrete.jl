
## state = [x, z, ẋ, ż]
let
    N = 4 # state dimension
    M = 2 # control dim
    h = 0.01 # time step size

    m = 10
    g = 9.81
    A = [0 0 1 0;
        0 0 0 1;
        0 0 0 0;
        0 0 0 0]

    B = [0  0;
        0  0;
        1/m  0;
        0  1/m]

    AB = [A B; zeros(M, N+M)]
    AB_exp = exp(AB*h)
    Ad = AB_exp[1:N, 1:N]
    Bd = AB_exp[1:N, N+1:N+M]
    G = [0, 0, 0, -g]
    println("Without gravity augmentation")
    println("Ad")
    display(Ad)
    println("Bd")
    display(Bd)
end 

## Gravity augmentation
# state = [x, z, ẋ, ż, g]
let
    N = 5
    M = 2
    h = 0.01

    m = 10
    g = 9.81
    A = [0 0 1 0 0;
        0 0 0 1 0;
        0 0 0 0 0;
        0 0 0 0 -1;
        0 0 0 0 0]

    B = [0 0;
        0 0;
        1/m 0;
        0 1/m;
        0 0]

    AB = [A B; zeros(M, N+M)]
    AB_exp = exp(AB)
    Ad = AB_exp[1:N, 1:N]
    Bd = AB_exp[1:N, N+1:N+M]

    println("With gravity augmentation")
    println("Ad")
    display(Ad)
    println("Bd")
    display(Bd)
end