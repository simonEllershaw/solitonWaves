# Soliton Grapher

This script was used in to analyse videos of the soliton wave phenomena
(for full details on how please see SolitionReport.pdf)

##To Run

Takes png input video frames with fnames as timestamps in ms. The locations 
is specified in the main function as 'directory'. Example data is given under the sample folder
directory. Two other parameters the yPixelperMeter 
and xPixelperMeter must also be specified.

Running solitionGrapher.py will then result in the graph of the data with the model fitted through and a txt with model 
parameters being outputted as: 'directory_graph.png' and 'directory_Parameters.txt'.