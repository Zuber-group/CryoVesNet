"""
Specifies parameters common to different individual tomogram (synapse)
processing scripts in order to make setting up multiple scripts easier.

This file is normally placed in another directory at the seme level as this 
on (e.g. ../common). Scripts that need to use parameters defined here need 
to import this script using:

  tomo_info = common.__import__(name='tomo_info', path='../common')
  
Each parameter should be read using a statement like:

  if tomo_info is not None: labels_file_name = tomo_info.labels_file_name

Parameters can have their values redefined in the individual scripts, if needed.

$Id: tomo_info.py 1573 2019-06-17 15:26:02Z vladan $
Author: Vladan Lucic 
"""
from __future__ import unicode_literals
from builtins import range

__version__ = "$Revision: 1573 $"


############################################################
#
# Image (grayscale) file. If it isn't in em or mrc format, format related
# variables (shape, data type, byte order and array order) need to be given
# (see labels file section below).
#

# name of the image file
image_file_name = "../3d/Dummy_133.rec.nad.rec"

###############################################################
#
# Labels file, specifies boundaries and possibly other regions. 
# 

# name of the labels file containing boundaries
labels_file_name = "../3d/labels_fixed.mrc"
max_sv_label = 491 #in practice, this is the maximum gray value in the labels file.

# labels file data type (e.g. 'int8', 'uint8', 'int16', 'int32', 'float16', 
# 'float64')
# For mrc and em files this should be set to None, otherwise it will override
# the data type specified in the header.
labels_data_type = 'int16'

#####################################################################
#
# Boundaries and other segments in labels
#

# Ids of all segments defined in the labels file (including boundaries).
# Segments that are not listed here are removed, that is set to 0. In order to
# avoid possible problems, all boundary file segments should be specified here, 
# or no other segment (boundary or segmentation region) should have id 0.
all_ids =  [1,2] + list(range(10,max_sv_label+1)) 

# Ids of all boundaries defined in the labels file. Nested list can be used 
# where ids in a sublist are uderstood in the "or" sense, that is all boundaries 
# listed in a sublist form effectivly a single boundary
boundary_ids = [2] + list(range(10,max_sv_label+1))

# Ids of vesicles in the labels file
vesicle_ids = list(range(10,max_sv_label+1))   

# Id of the segmentation region (where connectors can be formed). Using 0 in
# case the segmentation region is not specified in the boundary file is
# discouraged. 
segmentation_region = 1

# Id of a segment to which distance is calculated 
# For presynaptic_stats this should be the AZ membrane
distance_id = 2
