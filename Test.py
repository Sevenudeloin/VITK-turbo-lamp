import itk
import vtk
import numpy as np
from vtkmodules.util import numpy_support
import matplotlib.pyplot as plt

from visual_image import visual
from registration import registration

def segment_tumor_2d(slice_2d, seedX=125, seedY=70, lower=100, upper=255):
    normalize = itk.RescaleIntensityImageFilter.New(Input=slice_2d)
    normalize.SetOutputMinimum(0)
    normalize.SetOutputMaximum(255)
    normalize.Update()

    smoother = itk.GradientAnisotropicDiffusionImageFilter.New(Input=normalize.GetOutput(), NumberOfIterations=20,
                                                               TimeStep=0.04, ConductanceParameter=3)
    smoother.Update()

    connected_threshold = smoother
    connected_threshold = itk.ConnectedThresholdImageFilter.New(Input=connected_threshold)
    connected_threshold.SetReplaceValue(255)
    connected_threshold.SetLower(lower)
    connected_threshold.SetUpper(upper)
    connected_threshold.SetSeed((seedX, seedY))
    connected_threshold.Update()

    # plt.ion()
    # plt.imshow(connected_threshold.GetOutput(), cmap="gray")
    # plt.show()

    return connected_threshold.GetOutput()

def segment_tumor(input_image, output_path):
    slices = []
    for i in range(input_image.shape[0]):
        slice = itk.GetImageFromArray(itk.GetArrayViewFromImage(input_image)[i, :, :])
        seg_slice = segment_tumor_2d(slice)
        slices.append(itk.GetArrayViewFromImage(seg_slice))

    seg3d = itk.GetImageFromArray(np.stack(slices, axis=0))
    seg3d.SetSpacing(input_image.GetSpacing())
    seg3d.SetOrigin(input_image.GetOrigin())
    seg3d.SetDirection(input_image.GetDirection())

    print(seg3d.shape)
    itk.imwrite(seg3d, output_path)
    print("File Written")

if __name__ == "__main__":
    # print("registration")
    # registration()
    # print("end of registration")

    fixed_filepath = "./Data/case6_gre1_differenceafter.nrrd"
    moving_filepath = "./Data/case6_gre2_registered_rigid.nrrd"
    output_filepath = "./Data/result.nrrd"

    # fixed_image = itk.imread(fixed_filepath, pixel_type=itk.F)
    moving_image = itk.imread(moving_filepath, pixel_type=itk.F)
    # segment_tumor(fixed_image, output_filepath)
    segment_tumor(moving_image, output_filepath)
    visual(output_filepath)
    # visual(moving_filepath)

