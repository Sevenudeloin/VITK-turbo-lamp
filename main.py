import itk
import vtk
import numpy as np

# Load the fixed and moving images
def registration():
    fixed_image_path = './Data/case6_gre1.nrrd'
    moving_image_path = './Data/case6_gre2.nrrd'

    fixed_image = itk.imread(fixed_image_path, itk.F)
    moving_image = itk.imread(moving_image_path, itk.F)

    # Define the registration components
    Dimension = 3
    FixedImageType = itk.Image[itk.F, Dimension]
    MovingImageType = itk.Image[itk.F, Dimension]

    RigidTransformType = itk.VersorRigid3DTransform[itk.D]
    TransformInitializerType = itk.CenteredTransformInitializer[RigidTransformType, FixedImageType, MovingImageType]
    OptimizerType = itk.RegularStepGradientDescentOptimizer
    optimizer = OptimizerType.New()

    MetricType = itk.MattesMutualInformationImageToImageMetric[FixedImageType, MovingImageType]
    metric = MetricType.New()

    InterpolatorType = itk.LinearInterpolateImageFunction[MovingImageType, itk.D]
    interpolator = InterpolatorType.New()

    RegistrationType = itk.ImageRegistrationMethod[FixedImageType, MovingImageType]
    registration = RegistrationType.New()
    registration = RegistrationType.New()
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    registration.SetInterpolator(interpolator)
    IdentityTransformType = itk.IdentityTransform[itk.D, Dimension]
    identityTransform = IdentityTransformType.New()

    registration.SetFixedImage(fixed_image)
    registration.SetMovingImage(moving_image)
    metric.SetNumberOfHistogramBins(50)

    fixedRegion = fixed_image.GetBufferedRegion()
    numberOfPixels = fixedRegion.GetNumberOfPixels()
    metric.ReinitializeSeed(76926294)
    initializer = TransformInitializerType.New()
    rigidTransform = RigidTransformType.New()

    initializer.SetTransform(rigidTransform)
    initializer.SetFixedImage(fixed_image)
    initializer.SetMovingImage(moving_image)
    initializer.MomentsOn()
    initializer.InitializeTransform()
    registration.SetFixedImageRegion(fixedRegion)
    registration.SetInitialTransformParameters(rigidTransform.GetParameters())

    registration.SetTransform(rigidTransform)
    number_of_parameters = rigidTransform.GetNumberOfParameters()
    optimizerScales = [1.0] * number_of_parameters
    translationScale = 1.0 / 1000.0
    optimizerScales[3] = translationScale
    optimizerScales[4] = translationScale
    optimizerScales[5] = translationScale
    optimizer.SetScales(optimizerScales)

    optimizer.SetMaximumStepLength(0.2000)
    optimizer.SetMinimumStepLength(0.0001)

    optimizer.SetNumberOfIterations(200)
    metric.SetNumberOfSpatialSamples(10000)
    registration.Update()

    final_transform = RigidTransformType.New()
    final_transform.SetParameters(registration.GetLastTransformParameters())
    final_transform.SetFixedParameters(rigidTransform.GetFixedParameters())

    ResampleFilterType = itk.ResampleImageFilter[MovingImageType, FixedImageType]
    resampler = ResampleFilterType.New()
    resampler.SetTransform(final_transform)
    resampler.SetInput(moving_image)
    resampler.SetSize(fixed_image.GetLargestPossibleRegion().GetSize())
    resampler.SetOutputOrigin(fixed_image.GetOrigin())
    resampler.SetOutputSpacing(fixed_image.GetSpacing())
    resampler.SetOutputDirection(fixed_image.GetDirection())
    resampler.SetDefaultPixelValue(100)

    # Save the registered image
    output_image_path = './Data/case6_gre2_registered_rigid.nrrd'

    print("Registration completed successfully.")
    OutputPixelType = itk.US
    OutputImageType = itk.Image[OutputPixelType, Dimension]

    caster = itk.CastImageFilter[FixedImageType, OutputImageType].New()
    writer = itk.ImageFileWriter[OutputImageType].New()

    writer.SetFileName(output_image_path)
    caster.SetInput(resampler.GetOutput())
    writer.SetInput(caster.GetOutput())
    writer.Update()
    difference_filter = itk.SubtractImageFilter[FixedImageType, FixedImageType, FixedImageType].New()
    difference_filter.SetInput1(fixed_image)
    difference_filter.SetInput2(resampler.GetOutput())

    writer2 = itk.ImageFileWriter[OutputImageType].New()

    intensity_rescaler = itk.RescaleIntensityImageFilter[FixedImageType, OutputImageType].New()
    intensity_rescaler.SetInput(difference_filter.GetOutput())
    intensity_rescaler.SetOutputMinimum(0)
    intensity_rescaler.SetOutputMaximum(255)

    writer2.SetInput(intensity_rescaler.GetOutput())
    resampler.SetDefaultPixelValue(1)
    writer2.SetFileName("./Data/case6_gre2_differenceafter.nrrd")
    writer2.Update()

    # resampler.SetTransform(identityTransform)
    # writer2.SetFileName("./Data/case6_gre2_differencebefore.nrrd")
    # writer2.Update()

def segment_tumor_2d(slice_2d, seedX=125, seedY=65, lower=100, upper=255):
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

    # print(seg3d.shape)
    itk.imwrite(seg3d, output_path)
    print("Segmentation done")

def visualize(image1_path, image2_path):
    # === First image ===

    nrrdReader1 = vtk.vtkNrrdReader()
    nrrdReader1.SetFileName(image1_path)
    nrrdReader1.Update()

    # Calculate the center of the volume
    (xMin1, xMax1, yMin1, yMax1, zMin1, zMax1) = nrrdReader1.GetExecutive().GetWholeExtent(nrrdReader1.GetOutputInformation(0))
    (xSpacing1, ySpacing1, zSpacing1) = nrrdReader1.GetOutput().GetSpacing()
    (x01, y01, z01) = nrrdReader1.GetOutput().GetOrigin()

    center1 = [x01 + xSpacing1 * 0.5 * (xMin1 + xMax1),
            y01 + ySpacing1 * 0.5 * (yMin1 + yMax1),
            z01 + zSpacing1 * 0.5 * (zMin1 + zMax1)]

    # Matrices for sagittal, axial, coronal view orientations
    sagittal1 = vtk.vtkMatrix4x4()
    sagittal1.DeepCopy((1, 0, 0, center1[0],
                        0,-1, 0, center1[1],
                        0, 0, 1, center1[2],
                        0, 0, 0, 1))

    axial1 = vtk.vtkMatrix4x4()
    axial1.DeepCopy((0, 1, 0, center1[0],
                    0, 0, 1, center1[1],
                    1, 0, 0, center1[2],
                    0, 0, 0, 1))

    coronal1 = vtk.vtkMatrix4x4()
    coronal1.DeepCopy(( 0, 0, 1, center1[0],
                        0,-1, 0, center1[1],
                    -1, 0, 0, center1[2],
                        0, 0, 0, 1))

    # Extract a slice in the desired orientation
    reslice1 = vtk.vtkImageReslice()
    reslice1.SetInputConnection(nrrdReader1.GetOutputPort())
    reslice1.SetOutputDimensionality(2)
    reslice1.SetResliceAxes(sagittal1) # Change view here (axial, sagittal, coronal)
    reslice1.SetInterpolationModeToLinear()

    # Create a greyscale lookup table
    table1 = vtk.vtkLookupTable()
    table1.SetRange(0, 500) # image intensity range (0 1500)
    table1.SetValueRange(0.0, 1.0) # from black to white
    table1.SetSaturationRange(0.0, 0.0) # no color saturation
    table1.SetRampToLinear()
    table1.Build()

    # Map the image through the lookup table
    color1 = vtk.vtkImageMapToColors()
    color1.SetLookupTable(table1)
    color1.SetInputConnection(reslice1.GetOutputPort())

    # === Second image ===

    nrrdReader2 = vtk.vtkNrrdReader()
    nrrdReader2.SetFileName(image2_path)
    nrrdReader2.Update()

    # Calculate the center of the volume
    (xMin2, xMax2, yMin2, yMax2, zMin2, zMax2) = nrrdReader2.GetExecutive().GetWholeExtent(nrrdReader2.GetOutputInformation(0))
    (xSpacing2, ySpacing2, zSpacing2) = nrrdReader2.GetOutput().GetSpacing()
    (x02, y02, z02) = nrrdReader2.GetOutput().GetOrigin()

    center2 = [x02 + xSpacing2 * 0.5 * (xMin2 + xMax2),
            y02 + ySpacing2 * 0.5 * (yMin2 + yMax2),
            z02 + zSpacing2 * 0.5 * (zMin2 + zMax2)]

    # Matrices for sagittal, axial, coronal view orientations
    sagittal2 = vtk.vtkMatrix4x4()
    sagittal2.DeepCopy((1, 0, 0, center2[0],
                        0,-1, 0, center2[1],
                        0, 0, 1, center2[2],
                        0, 0, 0, 1))

    axial2 = vtk.vtkMatrix4x4()
    axial2.DeepCopy((0, 1, 0, center2[0],
                    0, 0, 1, center2[1],
                    1, 0, 0, center2[2],
                    0, 0, 0, 1))

    coronal2 = vtk.vtkMatrix4x4()
    coronal2.DeepCopy(( 0, 0, 1, center2[0],
                        0,-1, 0, center2[1],
                    -1, 0, 0, center2[2],
                        0, 0, 0, 1))

    # Extract a slice in the desired orientation
    reslice2 = vtk.vtkImageReslice()
    reslice2.SetInputConnection(nrrdReader2.GetOutputPort())
    reslice2.SetOutputDimensionality(2)
    reslice2.SetResliceAxes(sagittal2) # Change view here (axial, sagittal, coronal)
    reslice2.SetInterpolationModeToLinear()

    # Create a greyscale lookup table
    table2 = vtk.vtkLookupTable()
    table2.SetRange(0, 500) # image intensity range (0, 1500)
    table2.SetValueRange(0.0, 1.0) # from black to white
    table2.SetSaturationRange(0.0, 0.0) # no color saturation
    table2.SetRampToLinear()
    table2.Build()

    # Map the image through the lookup table
    color2 = vtk.vtkImageMapToColors()
    color2.SetLookupTable(table2)
    color2.SetInputConnection(reslice2.GetOutputPort())

    # === Display both images ===

    actor1 = vtk.vtkImageActor()
    actor1.GetMapper().SetInputConnection(color1.GetOutputPort())

    actor2 = vtk.vtkImageActor()
    actor2.GetMapper().SetInputConnection(color2.GetOutputPort())

    # Create a renderer for each image
    renderer1 = vtk.vtkRenderer()
    renderer2 = vtk.vtkRenderer()
    renderer1.AddActor(actor1)
    renderer2.AddActor(actor2)

    window = vtk.vtkRenderWindow()

    # First image to the left, second image to the right
    renderer1.SetViewport(0.0, 0.0, 0.5, 1.0)
    renderer2.SetViewport(0.5, 0.0, 1.0, 1.0)

    window.AddRenderer(renderer1)
    window.AddRenderer(renderer2)

    interactorStyle = vtk.vtkInteractorStyleImage()
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetInteractorStyle(interactorStyle)
    window.SetInteractor(interactor)
    window.Render()

    # Create callbacks for slicing the images
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
            
            reslice1.Update()
            reslice2.Update()
            
            sliceSpacing1 = reslice1.GetOutput().GetSpacing()[2]
            sliceSpacing2 = reslice2.GetOutput().GetSpacing()[2]
            
            matrix1 = reslice1.GetResliceAxes()
            matrix2 = reslice2.GetResliceAxes()
            
            # move the center point that we are slicing through
            center1 = matrix1.MultiplyPoint((0, 0, sliceSpacing1 * deltaY, 1))
            matrix1.SetElement(0, 3, center1[0])
            matrix1.SetElement(1, 3, center1[1])
            matrix1.SetElement(2, 3, center1[2])
            center2 = matrix2.MultiplyPoint((0, 0, sliceSpacing2 * deltaY, 1))
            matrix2.SetElement(0, 3, center2[0])
            matrix2.SetElement(1, 3, center2[1])
            matrix2.SetElement(2, 3, center2[2])
            
            window.Render()
        else:
            interactorStyle.OnMouseMove()

    interactorStyle.AddObserver("MouseMoveEvent", MouseMoveCallback)
    interactorStyle.AddObserver("LeftButtonPressEvent", ButtonCallback)
    interactorStyle.AddObserver("LeftButtonReleaseEvent", ButtonCallback)

    # Start interaction
    interactor.Start()

    # Clean up (useful?)
    del renderer1
    del renderer2
    del window
    del interactor

if __name__ == "__main__":
    # Registration
    print("Starting registration for second image")
    registration() # TODO leave on for final rendu

    # Base first image and registered second image
    fixed_filepath = "./Data/case6_gre1.nrrd"
    moving_filepath = "./Data/case6_gre2_registered_rigid.nrrd"

    # Segmentation result
    fixed_output_filepath = "./Data/case6_gre1_result.nrrd"
    moving_output_filepath = "./Data/case6_gre2_result.nrrd"

    # Segmentation
    fixed_image = itk.imread(fixed_filepath, pixel_type=itk.F)
    moving_image = itk.imread(moving_filepath, pixel_type=itk.F)
    print("Starting segmentation for first image")
    segment_tumor(fixed_image, fixed_output_filepath)
    print("Starting segmentation for second image")
    segment_tumor(moving_image, moving_output_filepath)

    # Visualisation
    print("Visualizing")
    visualize("./Data/case6_gre1_result.nrrd", "./Data/case6_gre2_result.nrrd")