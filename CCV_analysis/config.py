# Files' paths and names
SourceF = './Captured_video'                    # Source folder
ResF    = './CCV_analysis'                      # Result folder
tname   = 'Test1_Direct red'                    # Captured video name

# Printing parameters
ref_wdth = 0.428         # Nozzle tip diameter, unit: mm
inkwidth_trg = 0.285     # Target ink's width
Feed_rates = [90, 90]    # Feed rates [during printing, monitoring final print outcome]
Dye_selected = 'red'     # 'red' for Direct red, 'blue' for Brilliant blue dyes

#Video output
vid_res = 1280           # The video input resolution can be modified (e.g. width of 350, 740, 1000, up to 1280 px)
frame_skip = 0           # Skipping frames to process the video faster


#~~~~~~~~~~ Optional ~~~~~~~~~~#
# Custom thresholds frames
custom_range = False
Undistrupted_line = [0, 0]
Final_print = [0, 0]
Detection_range = [0, 0]