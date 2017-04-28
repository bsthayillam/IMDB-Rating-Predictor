import scipy

def gaussian(value, sigma, numUnits):
    # guassian curve equation 1/(sigma(sqrt(2pi)))e^(-1/2(x-mu/sigma)^2)
    mu = value
    binSize = 1./numUnits
    currentValue = binSize/2.

    activationValues = []*numUnits
    gaussianCurve = scipy.stats.norm(mu, sigma)
    for i in range(0, numUnits):
        val = gaussianCurve.pdf(currentValue)
        activationValues[i] = val
        currentValue = currentValue + binSize

    return activationValues



