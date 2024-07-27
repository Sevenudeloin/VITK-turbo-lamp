import itk

# class CommandIterationUpdate(itk.Command):
#     def __init__(self):
#         self.optimizer = None

#     def Execute(self, caller, event):
#         if not isinstance(caller, itk.RegularStepGradientDescentOptimizer):
#             return
        
#         optimizer = caller
#         if not itk.IterationEvent.CheckEvent(event):
#             return
        
#         print(f"{optimizer.GetCurrentIteration()}   "
#               f"{optimizer.GetValue()}   "
#               f"{optimizer.GetCurrentPosition()}")

def main():
    Dimension = 3
    PixelType = itk.F

    FixedImageType = itk.Image[PixelType, Dimension]
    MovingImageType = itk.Image[PixelType, Dimension]

    TransformType = itk.AffineTransform[itk.D, Dimension]
    OptimizerType = itk.RegularStepGradientDescentOptimizer
    MetricType = itk.MeanSquaresImageToImageMetric[FixedImageType, MovingImageType]
    InterpolatorType = itk.LinearInterpolateImageFunction[MovingImageType, itk.D]
    RegistrationType = itk.ImageRegistrationMethod[FixedImageType, MovingImageType]

    # Create and configure the registration components
    metric = MetricType.New()
    optimizer = OptimizerType.New()
    interpolator = InterpolatorType.New()
    registration = RegistrationType.New()

    transform = TransformType.New()
    registration.SetTransform(transform)
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    registration.SetInterpolator(interpolator)

    fixed_image_reader = itk.ImageFileReader[FixedImageType].New()
    moving_image_reader = itk.ImageFileReader[MovingImageType].New()
    fixed_image_reader.SetFileName("./Data/case6_gre1.nrrd")
    moving_image_reader.SetFileName("./Data/case6_gre2.nrrd")

    registration.SetFixedImage(fixed_image_reader.GetOutput())
    registration.SetMovingImage(moving_image_reader.GetOutput())
    fixed_image_reader.Update()

    registration.SetFixedImageRegion(fixed_image_reader.GetOutput().GetBufferedRegion())

    # mine
    RigidTransformType = itk.VersorRigid3DTransform[itk.D]
    rigid_transform = RigidTransformType.New()

    initializer = itk.CenteredTransformInitializer[RigidTransformType, FixedImageType, MovingImageType].New()
    initializer.SetTransform(rigid_transform)
    initializer.SetFixedImage(fixed_image_reader.GetOutput())
    initializer.SetMovingImage(moving_image_reader.GetOutput())
    initializer.MomentsOn()
    initializer.InitializeTransform()

    registration.SetInitialTransformParameters(transform.GetParameters())

    translation_scale = 1.0 / 1000.0 # float

    # mine
    num_parameters = transform.GetNumberOfParameters()

    optimizer_scales = itk.Array[itk.D](num_parameters)
    optimizer_scales.Fill(1.0)
    for i in range(9, num_parameters):
        optimizer_scales[i] = translation_scale
    optimizer.SetScales(optimizer_scales)

    step_length = 0.1 # float
    max_iterations = 700 # int

    optimizer.SetMaximumStepLength(step_length)
    optimizer.SetMinimumStepLength(0.0001)
    optimizer.SetNumberOfIterations(max_iterations)
    optimizer.MinimizeOn()

    # observer = CommandIterationUpdate()
    # optimizer.AddObserver(itk.IterationEvent(), observer)

    try:
        registration.Update()
        print("Optimizer stop condition: ", registration.GetOptimizer().GetStopConditionDescription())
    except itk.ITKException as e:
        print("Exception caught!")
        print(e)
        return 1

    final_parameters = registration.GetLastTransformParameters()
    number_of_iterations = optimizer.GetCurrentIteration()
    best_value = optimizer.GetValue()

    print("Result = ")
    print(f" Iterations    = {number_of_iterations}")
    print(f" Metric value  = {best_value}")

    ResampleFilterType = itk.ResampleImageFilter[MovingImageType, FixedImageType]
    final_transform = TransformType.New()
    final_transform.SetParameters(final_parameters)
    final_transform.SetFixedParameters(transform.GetFixedParameters())

    resampler = ResampleFilterType.New()
    resampler.SetTransform(final_transform)
    resampler.SetInput(moving_image_reader.GetOutput())

    fixed_image = fixed_image_reader.GetOutput()
    resampler.SetSize(fixed_image.GetLargestPossibleRegion().GetSize())
    resampler.SetOutputOrigin(fixed_image.GetOrigin())
    resampler.SetOutputSpacing(fixed_image.GetSpacing())
    resampler.SetOutputDirection(fixed_image.GetDirection())
    resampler.SetDefaultPixelValue(100)

    OutputPixelType = itk.UC
    OutputImageType = itk.Image[OutputPixelType, Dimension]

    caster = itk.CastImageFilter[FixedImageType, OutputImageType].New()
    writer = itk.ImageFileWriter[OutputImageType].New()

    writer.SetFileName("./Data/case6_gre2_registered.nrrd")
    caster.SetInput(resampler.GetOutput())
    writer.SetInput(caster.GetOutput())
    writer.Update()

    if False:
        difference_filter = itk.SubtractImageFilter[FixedImageType, FixedImageType, FixedImageType].New()
        difference_filter.SetInput1(fixed_image_reader.GetOutput())
        difference_filter.SetInput2(resampler.GetOutput())
        
        intensity_rescaler = itk.RescaleIntensityImageFilter[FixedImageType, OutputImageType].New()
        intensity_rescaler.SetInput(difference_filter.GetOutput())
        intensity_rescaler.SetOutputMinimum(0)
        intensity_rescaler.SetOutputMaximum(255)

        writer2 = itk.ImageFileWriter[OutputImageType].New()
        writer2.SetFileName("./Data/case6_gre2_differenceafter.nrrd")
        writer2.SetInput(intensity_rescaler.GetOutput())
        writer2.Update()

    if False:
        identity_transform = itk.IdentityTransform[itk.F, Dimension].New()
        resampler.SetTransform(identity_transform)
        writer2.SetFileName("./Data/case6_gre2_differencebefore.nrrd")
        writer2.Update()

    return 0

if __name__ == "__main__":
    main()
    # print(itk.CenteredTransformInitializer.GetTypes())
