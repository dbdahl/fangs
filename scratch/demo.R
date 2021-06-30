library(aibd)
library(fangs)

Zs <- aibd:::enumerateFeatureAllocations(6,3)

Z1 <- Zs[[33]]
Z2 <- Zs[[34]]

Z1
Z2

compute_loss(Z1,Z2)
compute_loss(Zs[[19]],Zs[[7]])

all <- sapply(Zs, function(Z) compute_loss(Zs[[19]],Z))
all[19] <- Inf
m <- min(all)
which(all==m)


fangs(Zs, 1, 1, -1, 100)


