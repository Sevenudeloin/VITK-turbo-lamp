import itk
import vtk
from vtkmodules.util import numpy_support
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

def visualize2D():
    brain_image1 = vtk.vtkNrrdReader()
    brain_image1.SetFileName("./Data/case6_gre1.nrrd")
    brain_image1.Update()

    brain_image2 = vtk.vtkNrrdReader()
    brain_image2.SetFileName("./Data/case6_gre2_registered_rigid.nrrd")
    brain_image2.Update()

    tumor_mask1 = vtk.vtkNrrdReader()
    tumor_mask1.SetFileName("./Data/case6_gre1_result.nrrd")
    tumor_mask1.Update()

    tumor_mask2 = vtk.vtkNrrdReader()
    tumor_mask2.SetFileName("./Data/case6_gre2_result.nrrd")
    tumor_mask2.Update()

    def clean_mask(vtk_mask):
        dims = vtk_mask.GetDimensions()
        scalars = vtk_mask.GetPointData().GetScalars()
        mask_np = np.reshape(numpy_support.vtk_to_numpy(scalars), dims, order='F')
        
        # Dirty fix for first and last slices
        mask_np[:, :, :30] = 0
        mask_np[:, :, 140:] = 0
        
        new_scalars = numpy_support.numpy_to_vtk(mask_np.ravel(order='F'), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_mask.GetPointData().SetScalars(new_scalars)
    clean_mask(tumor_mask1.GetOutput())
    clean_mask(tumor_mask2.GetOutput())

    brain_image1_total = np.count_nonzero(itk.imread("./Data/case6_gre1.nrrd"))
    brain_image2_total = np.count_nonzero(itk.imread("./Data/case6_gre2.nrrd"))

    tumor_mask1_np = itk.imread("./Data/case6_gre1_result.nrrd")
    tumor_mask1_np[:, :, :30] = 0
    tumor_mask1_np[:, :, 140:] = 0
    tumor_size1 = np.count_nonzero(tumor_mask1_np)

    tumor_mask2_np = itk.imread("./Data/case6_gre2_result.nrrd")
    tumor_mask2_np[:, :, :30] = 0
    tumor_mask2_np[:, :, 140:] = 0
    tumor_size2 = np.count_nonzero(tumor_mask2_np)

    (xMin, xMax, yMin, yMax, zMin, zMax) = brain_image1.GetExecutive().GetWholeExtent(brain_image1.GetOutputInformation(0))
    (xSpacing, ySpacing, zSpacing) = brain_image1.GetOutput().GetSpacing()
    (x0, y0, z0) = brain_image1.GetOutput().GetOrigin()

    center = [x0 + xSpacing * 0.5 * (xMin + xMax),
            y0 + ySpacing * 0.5 * (yMin + yMax),
            z0 + zSpacing * 0.5 * (zMin + zMax)]
    
    # Matrices for sagittal
    sagittal = vtk.vtkMatrix4x4()
    sagittal.DeepCopy((1, 0, 0, center[0],
                    0,-1, 0, center[1],
                    0, 0, 1, center[2],
                    0, 0, 0, 1))
    # Create renderers for each image
    renderer1 = vtk.vtkRenderer()
    renderer2 = vtk.vtkRenderer()

    # Create render window and interactor
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(800, 400)  # Set window size to accommodate side by side views

    render_window.AddRenderer(renderer1)
    render_window.AddRenderer(renderer2)

    # Set the viewport for side by side visualization
    renderer1.SetViewport(0.0, 0.0, 0.5, 1.0)
    renderer2.SetViewport(0.5, 0.0, 1.0, 1.0)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Function to create a lookup table to map mask values to red color
    def create_lookup_table():
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(2)
        lut.SetTableValue(0, 0.0, 0.0, 0.0, 0.0)  # Background: transparent
        lut.SetTableValue(1, 1.0, 0.0, 0.0, 0.7)  # Tumor: red with opacity
        lut.Build()
        return lut
    def lut_actor():
        lut = vtk.vtkLookupTable()
        lut.SetRange(0, 500)
        lut.SetValueRange(0.0, 1.0)
        lut.SetSaturationRange(0.0, 0.0) # no color saturation
        lut.SetRampToLinear()
        lut.Build()
        return lut
    reslice1 = vtk.vtkImageReslice()
    reslice1.SetInputData(brain_image1.GetOutput())
    reslice1.SetOutputDimensionality(2)
    reslice1.SetResliceAxes(sagittal) # Change view here
    reslice1.SetInterpolationModeToLinear()

    reslice1_mask = vtk.vtkImageReslice()
    reslice1_mask.SetInputData(tumor_mask1.GetOutput())
    reslice1_mask.SetOutputDimensionality(2)
    reslice1_mask.SetResliceAxes(sagittal) # Change view here
    reslice1_mask.SetInterpolationModeToLinear()

    color1 = vtk.vtkImageMapToColors()
    color1.SetLookupTable(lut_actor())
    color1.SetInputConnection(reslice1.GetOutputPort())

    # Create the image actor
    image_actor1 = vtk.vtkImageActor()
    image_actor1.GetMapper().SetInputConnection(color1.GetOutputPort())
    image_actor1.GetProperty().SetOpacity(0.5)  # Light white

    # Create a lookup table and map the mask to colors
    lut1 = create_lookup_table()
    map_to_colors1 = vtk.vtkImageMapToColors()
    map_to_colors1.SetInputData(reslice1_mask.GetOutput())
    map_to_colors1.SetLookupTable(lut1)
    map_to_colors1.Update()

    # Create the mask actor
    mask_actor1 = vtk.vtkImageActor()
    mask_actor1.GetMapper().SetInputConnection(map_to_colors1.GetOutputPort())

    # Add actors to the renderer
    renderer1.AddActor(image_actor1)
    renderer1.AddActor(mask_actor1)
    reslice2 = vtk.vtkImageReslice()
    reslice2.SetInputData(brain_image2.GetOutput())
    reslice2.SetOutputDimensionality(2)
    reslice2.SetResliceAxes(sagittal) # Change view here
    reslice2.SetInterpolationModeToLinear()

    reslice2_mask = vtk.vtkImageReslice()
    reslice2_mask.SetInputData(tumor_mask2.GetOutput())
    reslice2_mask.SetOutputDimensionality(2)
    reslice2_mask.SetResliceAxes(sagittal) # Change view here
    reslice2_mask.SetInterpolationModeToLinear()

    color2 = vtk.vtkImageMapToColors()
    color2.SetLookupTable(lut_actor())
    color2.SetInputConnection(reslice2.GetOutputPort())

    # Create the image actor
    image_actor2 = vtk.vtkImageActor()
    image_actor2.GetMapper().SetInputConnection(color2.GetOutputPort())
    image_actor2.GetProperty().SetOpacity(0.5)  # Light white

    # Create a lookup table and map the mask to colors
    lut2 = create_lookup_table()
    map_to_colors2 = vtk.vtkImageMapToColors()
    map_to_colors2.SetInputData(reslice2_mask.GetOutput())
    map_to_colors2.SetLookupTable(lut2)
    map_to_colors2.Update()

    # Create the mask actor
    mask_actor2 = vtk.vtkImageActor()
    mask_actor2.GetMapper().SetInputConnection(map_to_colors2.GetOutputPort())

    # Add actors to the renderer
    renderer2.AddActor(image_actor2)
    renderer2.AddActor(mask_actor2)
    cornerAnnotation1 = vtk.vtkCornerAnnotation()
    cornerAnnotation2 = vtk.vtkCornerAnnotation()

    renderer1.AddViewProp(cornerAnnotation1)
    renderer2.AddViewProp(cornerAnnotation2)
    interactorStyle = vtk.vtkInteractorStyleImage()
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetInteractorStyle(interactorStyle)
    render_window.SetInteractor(interactor)
    render_window.Render()

    cornerAnnotation1.SetText(2, f"Total size of the tumor: {int(tumor_size1)}px \n which is {(tumor_size1 * 100)/ brain_image1_total:.2f}% of the whole mri")
    cornerAnnotation2.SetText(2, f"Total size of the tumor: {int(tumor_size2)}px \n which is {(tumor_size2 * 100)/ brain_image2_total:.2f}% of the whole mri \n and is a difference of: {int(tumor_size2 * 100 / tumor_size1) - 100} %")
    # Create callbacks for slicing the image
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
            def update_slice(reslice):
                reslice.Update()
                sliceSpacing = reslice.GetOutput().GetSpacing()[2]
                matrix = reslice.GetResliceAxes()
                # move the center point that we are slicing through
                center = matrix.MultiplyPoint((0, 0, sliceSpacing * deltaY, 1))
                matrix.SetElement(0, 3, center[0])
                matrix.SetElement(1, 3, center[1])
                matrix.SetElement(2, 3, center[2])

            update_slice(reslice1)
            update_slice(reslice1_mask)
            update_slice(reslice2)
            update_slice(reslice2_mask)

            render_window.Render()
        else:
            interactorStyle.OnMouseMove()

    interactorStyle.AddObserver("MouseMoveEvent", MouseMoveCallback)
    interactorStyle.AddObserver("LeftButtonPressEvent", ButtonCallback)
    interactorStyle.AddObserver("LeftButtonReleaseEvent", ButtonCallback)
    # Start interaction
    interactor.Start()
    del render_window
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
    visualize2D()