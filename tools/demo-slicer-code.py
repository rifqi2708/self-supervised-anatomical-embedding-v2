# Slicer manupulating fiducial points
# https://slicer.readthedocs.io/en/latest/developer_guide/script_repository.html#access-to-markups-point-list-properties
# Under data get the node name. Multiple control points can sit under one node
import numpy as np
from pathlib import Path
import time

# Step 1 - Slicer code create window

layoutName = "TestSlice1"
layoutLabel = "TS1"
layoutColor = [1.0, 1.0, 0.0]
# ownerNode manages this view instead of the layout manager (it can be any node in the scene)
viewOwnerNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScriptedModuleNode")

# Create MRML nodes
viewLogic = slicer.vtkMRMLSliceLogic()
viewLogic.SetMRMLScene(slicer.mrmlScene)
viewNode = viewLogic.AddSliceNode(layoutName)
viewNode.SetLayoutLabel(layoutLabel)
viewNode.SetLayoutColor(layoutColor)
viewNode.SetAndObserveParentLayoutNodeID(viewOwnerNode.GetID())

viewNode.SetOrientation('Axial') # Show in Axial direction

# Create widget
viewWidget = slicer.qMRMLSliceWidget()
viewWidget.setMRMLScene(slicer.mrmlScene)
viewWidget.setMRMLSliceNode(viewNode)
sliceLogics = slicer.app.applicationLogic().GetSliceLogics()
viewWidget.setSliceLogics(sliceLogics)
sliceLogics.AddItem(viewWidget.sliceLogic())
viewWidget.show()


# # Step 2- Create a Fiducial point (Markup module) named "F", create two points under F

# # Step 3 - Get the node
# n = getNode('F')
# n.GetNumberOfControlPoints()  # Number of points
# n.GetNthControlPointLabel(1)  # n-th label
# n.SetNthControlPointPosition(0, (194.87, 71.87, -730.6))

# Fiducial points
def add_fiducial_pt():
    node_id = slicer.modules.markups.logic().AddNewFiducialNode()
    node = getNode(node_id)
    pt_id_1 = node.AddControlPoint(0,0,0)
    pt_id_2 = node.AddControlPoint(0,0,0)
    return node
n = add_fiducial_pt()

# Define the Mapper class

class Mapper():
    def __init__(self, node):
        self.node = node
        self.matrix = self.getMatrix(self.node)
        self.matrix_inv = np.linalg.inv(self.matrix)
        
    @staticmethod
    def getMatrix(node):
        ori = node.GetOrigin()
        spa = node.GetSpacing()
        
        matrix = np.zeros((4,4), np.float32)
        matrix[0,0] = -spa[0]
        matrix[1,1] = -spa[1]
        matrix[2,2] = spa[2]
        matrix[3,3] = 1

        matrix[:3, 3] = ori

        return matrix

    @staticmethod
    def applyMatrix(matrix, coord):
        if len(coord) != 4:
            coord = np.array([*coord, 1])
        return (matrix @ coord)[:3]   
    
    def ijk2ras(self, *coord):
        return self.applyMatrix(self.matrix, coord)

    def ras2ijk(self, *coord):
        return self.applyMatrix(self.matrix_inv, coord).astype(int)
    

# Load the data
# Data (.nii) under P:\yusun\self-supervised-anatomical-embedding-v2\data\raw_data\NIH_lymph_node
data_dir = Path(r'P:\yusun\self-supervised-anatomical-embedding-v2\data\raw_data\NIH_lymph_node')
img1 = loadVolume(data_dir/'ABD_LYMPH_001.nii.gz')
img2 = loadVolume(data_dir/'ABD_LYMPH_002.nii.gz')

mapper1 = Mapper(img1)
mapper2 = Mapper(img2)



# Get the ijk of the F-1
# Then send to the model to get the correspondence - e.g. returns [ 86, 112,  48]
mapper1.ras2ijk(*n.GetNthControlPointPosition(0))

# Update F-2
n.SetNthControlPointPosition(1, mapper2.ijk2ras(*[ 86, 112,  48]))

import time
from pathlib import Path

file1 = Path(r'P:\yusun\self-supervised-anatomical-embedding-v2\tools\assets\demo\f-1.txt')
file2 = Path(r'P:\yusun\self-supervised-anatomical-embedding-v2\tools\assets\demo\f-2.txt')
f1c = eval(open(file1).read())
f2c = eval(open(file2).read())

# def refresh():
#     f1s = mapper1.ras2ijk(*n.GetNthControlPointPosition(0)).tolist()
#     f1c = eval(open(file1).read())
#     if f1s == f1c: return
#     else:
#         print(f1s, file=open(file1, 'w'))
#         time.sleep(10)
#         f2c = eval(open(file2).read())
#         n.SetNthControlPointPosition(1, mapper2.ijk2ras(*f2c))


def make_progress_bar(period=10):
    progressbar=slicer.util.createProgressDialog(parent=slicer.util.mainWindow(),windowTitle='Processing...',autoClose=True)
    for i in range(period):
        time.sleep(1)
        progressbar.value += int(100 / period)
    progressbar.cancel()

def wait(message, period=10):
    slicer.util.delayDisplay(message, autoCloseMsec=period*1000)

# Event listener for markup end of interaction (e.g. finish dragging)
@vtk.calldata_type(vtk.VTK_INT)
def callback(node, event):
    f1s = mapper1.ras2ijk(*n.GetNthControlPointPosition(0)).tolist() # Fiducial point reading
    f1c = eval(open(file1).read()) # Current reading
    if f1s == f1c: return
    else:
        print(f1s, file=open(file1, 'w'))

        with slicer.util.WaitCursor():
            time_lapse = 0
            while True:
                time.sleep(1)
                time_lapse += 1
                f2c = eval(open(file2).read())
                
                if f2c['pt1'] == f1s: break
                if time_lapse > 15: 
                    print('Time out. No updates')
                    return
        pt2 = f2c['pt2']

        coord = mapper2.ijk2ras(*pt2)   # Absolute scale
        node.SetNthControlPointPosition(1, coord)   # Update pt position
        print(pt2, mapper2.ras2ijk(*n.GetNthControlPointPosition(1)))
        print(f'Coord: {coord}. Jumped to {coord[-1]}')
        viewLogic.SetSliceOffset(coord[-1]) # Jump to the slice
    
obid = n.AddObserver(slicer.vtkMRMLMarkupsFiducialNode.PointEndInteractionEvent, callback)

# De-register
n.RemoveObserver(obid)

# vtkMRMLMarkupsFiducialNode events
['CenterOfRotationModifiedEvent',
 'DisplayModifiedEvent',
 'FixedNumberOfControlPointsModifiedEvent',
 'GetContentModifiedEvents',
 'GetCustomModifiedEventPending',
 'GetDisableModifiedEvent',
 'GetModifiedEventPending',
 'HierarchyModifiedEvent',
 'IDChangedEvent',
 'InvokeCustomModifiedEvent',
 'InvokeEvent',
 'InvokePendingModifiedEvent',
 'LabelFormatModifiedEvent',
 'LockModifiedEvent',
 'PointAboutToBeRemovedEvent',
 'PointAddedEvent',
 'PointEndInteractionEvent',
 'PointModifiedEvent',
 'PointPositionDefinedEvent',
 'PointPositionMissingEvent',
 'PointPositionNonMissingEvent',
 'PointPositionUndefinedEvent',
 'PointRemovedEvent',
 'PointStartInteractionEvent',
 'ProcessMRMLEvents',
 'ReferenceAddedEvent',
 'ReferenceModifiedEvent',
 'ReferenceRemovedEvent',
 'ReferencedNodeModifiedEvent',
 'SetDisableModifiedEvent',
 'TransformModifiedEvent']

