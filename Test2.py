import itk
import vtk
import numpy as np
from vtkmodules.util import numpy_support

if __name__ == "__main__":
    fixed_filepath = "./Data/case6_gre1.nrrd"
    moving_filepath = "./Data/case6_gre2.nrrd"
    PixelType = itk.US
    fixed_image = itk.imread(fixed_filepath, PixelType)
    moving_image = itk.imread(moving_filepath, PixelType)
    InputPixelType = PixelType
    OutputPixelType = itk.F

    InputImageType = itk.Image[InputPixelType, 3]
    OutputImageType = itk.Image[OutputPixelType, 3]

    CastFilterType = itk.CastImageFilter[InputImageType, OutputImageType]
    WriterType = itk.ImageFileWriter[OutputImageType]

    castFilter01 = CastFilterType.New()
    writer01 = WriterType.New()

    castFilter01.SetInput(fixed_image)
    castFilter01.Update()

    fixed_f_filepath = "./Data/case6_gre1_f.nrrd"
    writer01.SetFileName(fixed_f_filepath)
    writer01.SetInput(castFilter01.GetOutput())
    writer01.Update()

    castFilter02 = CastFilterType.New()
    writer02 = WriterType.New()

    castFilter02.SetInput(moving_image)
    castFilter02.Update()

    moving_f_filepath = "./Data/case6_gre2_f.nrrd"
    writer02.SetFileName(moving_f_filepath)
    writer02.SetInput(castFilter02.GetOutput())
    writer02.Update()
