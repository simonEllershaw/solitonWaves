import numpy as np
import matplotlib.pyplot as pyplot
import cv2
from scipy.optimize import curve_fit
import os
from itertools import cycle
import warnings


def analyseDirectory(directory, plotGraph, calcModelParameters):
    n = []
    h = []
    L = []
    p = []
    time = []
    chiSquaredValues = []
    colCycle = cycle('bgrcm')

    for filename in os.listdir(directory):
        x, imageProfile, modelParams, residuals = analyseInputImage(filename, n, h, L, p, time, chiSquaredValues)
        #  Plot experimental and then model data to visualise fit
        #  Chi squared needs
        if plotGraph:
            timeColour = next(colCycle)
            dataPlot.plot(x, imageProfile, 'x', label="%dms" % time[-1],
                          color=timeColour)  # cycle dif color for each plot
            dataPlot.plot(x, modelSoliton(x, *modelParams), color='k', linewidth=2.0)  # model fit black line
            # Plot residuals in same colour
            residualPlot.plot(x, residuals, linestyle="none", marker="x", color=timeColour)
    if calcModelParameters:
        calcAndWriteParametersToFile(directory, n, h, L, p, time, chiSquaredValues)


def analyseInputImage(filename, n, h, L, p, time, chiSquaredValues):
    #  Read in file and extract wave profile and parameters through thresholding
    filepath, timeStamp = getFilePathAndTimeStamp(filename)
    image = cv2.imread(filepath)
    imageProfile = imageToProfileArray(image) / yPixelperMeter
    x = np.arange(len(imageProfile)) / xPixelperMeter
    profile_yerr = (np.zeros(len(x)) + 2) / yPixelperMeter
    modelParams, modelParams_error = modelFit(imageProfile, profile_yerr, x)
    #  Append parameters to arrays
    n.append(modelParams[0])
    h.append(modelParams[1])
    L.append(modelParams[2])
    p.append(modelParams[3])
    time.append(timeStamp)
    #  Calculate residuals and chi-squared values
    residuals = (imageProfile - modelSoliton(x, *modelParams)) / profile_yerr
    chiSquaredValues.append(sum(((imageProfile - modelSoliton(x, *modelParams)) / profile_yerr) ** 2))
    return x, imageProfile, modelParams, residuals


def getFilePathAndTimeStamp(filename):
    thisFilePath = directory + "/" + filename
    #  Recording device gives filenames as timestamps
    thisTimeStamp = int(os.path.splitext(filename)[0])
    return thisFilePath, thisTimeStamp


def imageToProfileArray(inputImage):
    #  Take blue channel and apply thresholding and sum columns to get wave profile in pixels
    blueChannel = inputImage[:, :, 2]
    for i in range(len(blueChannel)):
        for j in range(len(blueChannel[i])):
            if (blueChannel[i][j] < 100):  # 100 found to arbitrarily effective threshold
                blueChannel[i][j] = 1
            else:
                blueChannel[i][j] = 0
    #  Sum vertically to get wave profile
    verticalIntensities = np.sum(blueChannel, axis=0)
    return verticalIntensities


def modelSoliton(x, n, h, L, p):
    #  As given by Bettini et al. link to paper in ReadMe.txt
    #  n amplitude, h depth of tank, L characteristic length of wave and p the peak position
    #  measured from the LHS of the frame
    return n * (1 / (np.cosh((x - p) / L))) ** 2 + h


def modelFit(profile, err, x):
    #  Fit model and experimental data. Returns values of parameters and their errors
    params, cov_matrix = curve_fit(modelSoliton, x, profile, sigma=err, absolute_sigma=True, p0=[0.2, 0.05, 0.12, 0.15],
                                   maxfev=1000000000)
    errors = np.sqrt(np.diag(cov_matrix))
    return params, errors


def calcAndWriteParametersToFile(directory, n, h, L, p, time, chiSquaredValues):
    paramFile = open("%s_Parameters.txt" % directory, "w")
    paramFile.write("File: %s\n" % directory)

    timeConverted = [i / 1000 for i in time]  # Convert from ms to s
    numFiles = len(os.listdir(directory))

    paramFile.write("\nChi Squared Values\n")
    writeChiSquaredValuesToFile(timeConverted, chiSquaredValues, paramFile)

    paramFile.write("\nWave Parameters\n")
    calcMeanAndStdError("n", "m", n, numFiles, paramFile)
    calcMeanAndStdError("h", "m", h, numFiles, paramFile)
    calcMeanAndStdError("L", "m^2", L, numFiles, paramFile)

    speed = calcWaveSpeedValues(timeConverted, p)
    calcMeanAndStdError("Speed", "ms^-1", speed, numFiles, paramFile)
    paramFile.close()


def writeChiSquaredValuesToFile(timeConverted, chiSquaredValues, openFile):
    for i in range(len(chiSquaredValues)):
        openFile.write("%.3fs: %f\n" % (timeConverted[i], chiSquaredValues[i]))


def calcWaveSpeedValues(time, p):
    #  Using adjacent timestamps and peak positions the speed of the wave can be calculated
    speed = []
    for i in range(len(p) - 1):
        speedValue = abs((p[i + 1] - p[i]) / (time[i + 1] - time[i]))
        speed.append(speedValue)
    return speed


def calcMeanAndStdError(paramName, paramUnits, paramValues, numFiles, openFile):
    avParam = np.sum(paramValues) / numFiles
    stdErrorParam = np.std(paramValues) / np.sqrt(numFiles)
    openFile.write("%s = (%f ± %f)%s\n" % (paramName, avParam, stdErrorParam, paramUnits))
    # return avParam, stdErrorParam


def setupGraph():
    fig = pyplot.figure()
    fig.set_size_inches(10, 10)
    dataPlot = fig.add_axes((.1, .3, .8, .6))
    residualPlot = fig.add_axes((.1, .1, .8, .2))
    return fig, dataPlot, residualPlot


def formatGraph(dataPlot, residualPlot, fig, directory):
    #  format data plot
    dataPlot.tick_params(axis="y", direction="in")
    dataPlot.tick_params(axis="x", bottom=False)
    dataPlot.autoscale(axis="x", tight=True)
    dataPlot.autoscale(axis="y", tight=False)
    dataPlot.set_ylabel("Y Displacement / m", fontSize=12)
    #  format residual plot
    residualPlot.tick_params(axis="y", direction="in")
    residualPlot.tick_params(axis="x", direction="in")
    residualPlot.fill_between([0, 0.32], 1, -1, color="#C0C0C0")
    residualPlot.autoscale(axis="x", tight=True)
    residualPlot.autoscale(axis="y", tight=False)
    residualPlot.set_xlabel("X Displacement / m", fontSize=12)
    residualPlot.set_ylabel("Normalised Residuals", fontSize=12)
    #  Show legend and figure
    fig.legend()
    fig.savefig(directory+"_graph.png")
    fig.show()


#  Does not seem to effect output
warnings.filterwarnings(
    action='ignore', module='matplotlib.figure', category=UserWarning,
    message=('This figure includes Axes that are not compatible with tight_layout, '
             'so results might be incorrect.')
)

if __name__ == "__main__":
    #  User dependent input
    directory = "sampleData"
    #   Measured empirically
    yPixelperMeter = 2827
    xPixelperMeter = 2170
    plotGraph = True
    calcModelParameters = True  # Parameters saved in file named directoryName_Parameters.txt
    #  Automated analysis work
    if plotGraph:
        fig, dataPlot, residualPlot = setupGraph()
    analyseDirectory(directory, plotGraph, calcModelParameters)
    if plotGraph:
        formatGraph(dataPlot, residualPlot, fig, directory)
