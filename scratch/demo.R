library(fangs)

data(samplesFA)
samplesFA

fangs(samplesFA, nIterations = 0, maxSeconds = 0, algorithm = "neighbors")
