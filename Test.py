import itk
import vtk
import numpy as np
from vtkmodules.util import numpy_support
import matplotlib.pyplot as plt

# import TODO

def visual(file_name):
    nrrdReader = vtk.vtkNrrdReader()
    nrrdReader.SetFileName(file_name)
    nrrdReader.Update()
    nrrdReader.Update()
    (xMin, xMax, yMin, yMax, zMin, zMax) = nrrdReader.GetExecutive().GetWholeExtent(nrrdReader.GetOutputInformation(0))
    (xSpacing, ySpacing, zSpacing) = nrrdReader.GetOutput().GetSpacing()
    (x0, y0, z0) = nrrdReader.GetOutput().GetOrigin()

    center = [x0 + xSpacing * 0.5 * (xMin + xMax),
              y0 + ySpacing * 0.5 * (yMin + yMax),
              z0 + zSpacing * 0.5 * (zMin + zMax)]
    axial = vtk.vtkMatrix4x4()
    axial.DeepCopy((1, 0, 0, center[0],
                    0, 1, 0, center[1],
                    0, 0, 1, center[2],
                    0, 0, 0, 1))

    coronal = vtk.vtkMatrix4x4()
    coronal.DeepCopy((1, 0, 0, center[0],
                      0, 0, 1, center[1],
                      0, -1, 0, center[2],
                      0, 0, 0, 1))

    sagittal = vtk.vtkMatrix4x4()
    sagittal.DeepCopy((0, 0, -1, center[0],
                       1, 0, 0, center[1],
                       0, -1, 0, center[2],
                       0, 0, 0, 1))

    oblique = vtk.vtkMatrix4x4()
    oblique.DeepCopy((1, 0, 0, center[0],
                      0, 0.866025, -0.5, center[1],
                      0, 0.5, 0.866025, center[2],
                      0, 0, 0, 1))

    reslice = vtk.vtkImageReslice()
    reslice.SetInputConnection(nrrdReader.GetOutputPort())
    reslice.SetOutputDimensionality(2)
    reslice.SetResliceAxes(axial)
    reslice.SetInterpolationModeToLinear()

    table = vtk.vtkLookupTable()
    table.SetRange(0, 2000)
    table.SetValueRange(0.0, 1.0)
    table.SetSaturationRange(0.0, 0.0)
    table.SetRampToLinear()
    table.Build()

    color = vtk.vtkImageMapToColors()
    color.SetLookupTable(table)
    color.SetInputConnection(reslice.GetOutputPort())

    actor = vtk.vtkImageActor()
    actor.GetMapper().SetInputConnection(color.GetOutputPort())

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)

    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)

    interactorStyle = vtk.vtkInteractorStyleImage()
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetInteractorStyle(interactorStyle)
    window.SetInteractor(interactor)
    window.Render()

    actions = {}
    actions["Slicing"] = 0

    def ButtonCallback(obj, event):
        if event == "LeftButtonPressEvent":
            actions["Slicing"] = 1
        else:
            actions["Slicing"] = 0

    def MouseMoveCallback(obj, event):
        (lastX, lastY) = interactor.GetLastEventPosition()
        (mouseX, mouseY) = interactor.GetEventPosition()
        if actions["Slicing"] == 1:
            deltaY = mouseY - lastY
            reslice.Update()
            sliceSpacing = reslice.GetOutput().GetSpacing()[2]
            matrix = reslice.GetResliceAxes()
            # move the center point that we are slicing through
            center = matrix.MultiplyPoint((0, 0, sliceSpacing * deltaY, 1))
            matrix.SetElement(0, 3, center[0])
            matrix.SetElement(1, 3, center[1])
            matrix.SetElement(2, 3, center[2])
            window.Render()
        else:
            interactorStyle.OnMouseMove()

    interactorStyle.AddObserver("MouseMoveEvent", MouseMoveCallback)
    interactorStyle.AddObserver("LeftButtonPressEvent", ButtonCallback)
    interactorStyle.AddObserver("LeftButtonReleaseEvent", ButtonCallback)

    interactor.Start()
    del renderer
    del window
    del interactor

def segment_tumor(image):
    # otsu_filter = itk.OtsuThresholdImageFilter.New(Input=image, InsideValue=1, OutsideValue=0)
    # otsu_filter.Update()
    # thresholded_image = otsu_filter.GetOutput()
    #
    # connected_filter = itk.ConnectedComponentImageFilter.New(Input=thresholded_image)
    # connected_filter.Update()
    # connected_image = connected_filter.GetOutput()
    #
    # relabel_filter = itk.RelabelComponentImageFilter.New(Input=connected_image)
    # relabel_filter.Update()
    smoother = itk.GradientAnisotropicDiffusionImageFilter.New(Input=image, NumberOfIterations=20, TimeStep=0.04,
                                                               ConductanceParameter=3)
    smoother.Update()
    connected_threshold = itk.ConnectedThresholdImageFilter.New(smoother.GetOutput())
    connected_threshold.SetReplaceValue(255)
    connected_threshold.SetLower(150)
    connected_threshold.SetUpper(255)
    connected_threshold.Update()
    return connected_threshold.GetOutput()

if __name__ == "__main__":
    fixed_filepath = "./Data/case6_gre1.nrrd"
    moving_filepath = "./Data/case6_gre2.nrrd"

    # test_image = itk.imread('./Data/brain.png', pixel_type=itk.F)
    test_image = itk.imread('./Data/brain2.png', pixel_type=itk.F)

    InputPixelType = itk.US
    OutputPixelType = itk.F

    InputImageType = itk.Image[InputPixelType, 3]
    OutputImageType = itk.Image[OutputPixelType, 3]

    # fixed_image = itk.imread(fixed_filepath, pixel_type=itk.US)
    # test_image = fixed_image[50, :, :]

    # # moving_image = itk.imread(moving_filepath, PixelType)
    # plt.ion()
    # plt.imshow(fixed_image, cmap="gray")
    # plt.show()
    #
    # print(fixed_image.shape)
    # fixed_segmented = segment_tumor(fixed_image)
    # print(fixed_segmented.shape)
    # # moving_segmented = segment_tumor(moving_image)
    # plt.ion()
    # plt.imshow(fixed_segmented, cmap="gray")
    # plt.show()

    # plt.ion()
    # plt.imshow(moving_segmented, cmap="gray")
    # plt.show()
    # visual("./Data/case6_gre1.nrrd")

    smoother = itk.GradientAnisotropicDiffusionImageFilter.New(Input=test_image,
                                                                NumberOfIterations=20, TimeStep=0.04, ConductanceParameter=3)
    smoother.Update()

    connected_threshold = itk.ConnectedThresholdImageFilter.New(smoother.GetOutput())
    connected_threshold.SetReplaceValue(255)
    connected_threshold.SetLower(100)
    connected_threshold.SetUpper(255)

    connected_threshold.SetSeed((110, 100))
    connected_threshold.Update()
    plt.imshow(connected_threshold.GetOutput(), cmap="gray")
    plt.show()
