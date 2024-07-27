import itk
import vtk
import numpy as np
from vtkmodules.util import numpy_support
import matplotlib.pyplot as plt

from visual_image import visual
from registration import registration

def segment_tumor_2d(slice_2d, seedX=110, seedY=100, lower=180, upper=255):
    smoother = itk.GradientAnisotropicDiffusionImageFilter.New(Input=slice_2d, NumberOfIterations=20, TimeStep=0.04,
                                                               ConductanceParameter=3)
    smoother.Update()

    connected_threshold = smoother
    connected_threshold = itk.ConnectedThresholdImageFilter.New(Input=connected_threshold)
    connected_threshold.SetReplaceValue(255)
    connected_threshold.SetLower(lower)
    connected_threshold.SetUpper(upper)
    connected_threshold.SetSeed((seedX, seedY))
    connected_threshold.Update()

    # in_type = itk.output(connected_threshold)
    # output_type = itk.Image[itk.UC, slice_2d.GetImageDimension()]
    # rescaler = itk.RescaleIntensityImageFilter[in_type, output_type].New(Input=connected_threshold.GetOutput())
    # rescaler.SetOutputMinimum(0)
    # rescaler.SetOutputMaximum(255)
    # rescaler.Update()

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
    print("registration")
    registration()
    print("end of registration")

    fixed_filepath = "./Data/case6_gre1.nrrd"
    moving_filepath = "./Data/case6_gre2_registered_rigid.nrrd"
    output_filepath = "./Data/case6_gre1_seg.nrrd"
    output_filepath2 = "./Data/case6_gre2_seg.nrrd"

    fixed_image = itk.imread(fixed_filepath, pixel_type=itk.F)
    moving_image = itk.imread(moving_filepath, pixel_type=itk.F)
    segment_tumor(fixed_image, output_filepath)
    segment_tumor(moving_image, output_filepath2)

    # visual(output_filepath)
    # visual(moving_filepath)

