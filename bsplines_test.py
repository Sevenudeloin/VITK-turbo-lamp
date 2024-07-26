import itk
import vtk

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

    # =====

    ImageDimension = 3
    PixelType = itk.F # itk.ctype("float")
    
    FixedImageType = itk.Image[PixelType, ImageDimension]
    MovingImageType = itk.Image[PixelType, ImageDimension]
    SpaceDimension = ImageDimension
    SplineOrder = 3
    CoordinateRepType = itk.D # itk.ctype("double")
    
    TransformType = itk.BSplineTransform[CoordinateRepType, SpaceDimension, SplineOrder]
    OptimizerType = itk.LBFGSBOptimizerv4
    MetricType = itk.MeanSquaresImageToImageMetricv4[FixedImageType, MovingImageType]
    RegistrationType = itk.ImageRegistrationMethodv4[FixedImageType, MovingImageType]

    metric = MetricType.New()
    optimizer = OptimizerType.New()
    registration = RegistrationType.New()
    
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    FixedImageReaderType = itk.ImageFileReader[FixedImageType]
    MovingImageReaderType = itk.ImageFileReader[MovingImageType]

    fixedImageReader = FixedImageReaderType.New()
    movingImageReader = MovingImageReaderType.New()

    #fixed_filepath = "./Data/case6_gre1.nrrd"
    #moving_filepath = "./Data/case6_gre2.nrrd"
    fixed_filepath = "./Data/case6_gre1_f.nrrd"
    moving_filepath = "./Data/case6_gre2_f.nrrd"
    fixedImageReader.SetFileName(fixed_filepath)
    movingImageReader.SetFileName(moving_filepath)

    fixedImage = fixedImageReader.GetOutput()

    registration.SetFixedImage(fixedImage)
    registration.SetMovingImage(movingImageReader.GetOutput())

    fixedImageReader.Update()
    outputBSplineTransform = TransformType.New()

    # Initialize the transform
    InitializerType = itk.BSplineTransformInitializer[TransformType, FixedImageType]

    transformInitializer = InitializerType.New()

    numberOfGridNodesInOneDimension = 8

    meshSize = itk.Size[ImageDimension]()
    meshSize.Fill(numberOfGridNodesInOneDimension - SplineOrder)

    transformInitializer.SetTransform(outputBSplineTransform)
    transformInitializer.SetImage(fixedImage)
    transformInitializer.SetTransformDomainMeshSize(meshSize)
    transformInitializer.InitializeTransform()

    # Set transform to identity
    ParametersType = itk.OptimizerParameters[CoordinateRepType]
    numberOfParameters = outputBSplineTransform.GetNumberOfParameters()
    parameters = ParametersType(numberOfParameters)
    parameters.Fill(0.0)
    outputBSplineTransform.SetParameters(parameters)
    registration.SetInitialTransform(outputBSplineTransform)
    registration.InPlaceOn()

    numberOfLevels = 1

    shrinkFactorsPerLevel = itk.Array[itk.UL]() # FIXME maybe type
    shrinkFactorsPerLevel.SetSize(numberOfLevels)
    shrinkFactorsPerLevel[0] = 1

    smoothingSigmasPerLevel = itk.Array[itk.D]() # FIXME maybe type
    smoothingSigmasPerLevel.SetSize(numberOfLevels)
    smoothingSigmasPerLevel[0] = 0

    registration.SetNumberOfLevels(numberOfLevels)
    registration.SetSmoothingSigmasPerLevel(smoothingSigmasPerLevel)
    registration.SetShrinkFactorsPerLevel(shrinkFactorsPerLevel)
    numParameters = outputBSplineTransform.GetNumberOfParameters()
    BoundSelectionType = itk.Array[itk.UL] # FIXME ? type
    boundSelect = BoundSelectionType()
    boundSelect.SetSize(numParameters)

    BoundValueType = itk.Array[itk.D] # FIXME ? type
    upperBound = BoundValueType()
    upperBound.SetSize(numParameters)
    lowerBound = BoundValueType()
    lowerBound.SetSize(numParameters)

    UNBOUNDED = 0 # because OptimizerType = itk.LBFGSBOptimizerv4
    boundSelect.Fill(UNBOUNDED)
    upperBound.Fill(0.0)
    lowerBound.Fill(0.0)

    optimizer.SetBoundSelection(boundSelect)
    optimizer.SetUpperBound(upperBound)
    optimizer.SetLowerBound(lowerBound)

    optimizer.SetCostFunctionConvergenceFactor(1e+12)
    optimizer.SetGradientConvergenceTolerance(1.0e-35)
    optimizer.SetNumberOfIterations(500)
    optimizer.SetMaximumNumberOfFunctionEvaluations(500)
    optimizer.SetMaximumNumberOfCorrections(5)

    print("ici", flush=True)

    try:
        registration.Update()
        print(registration.GetOptimizer().GetStopConditionDescription(), flush=True)
    except Exception as err:
        print(err, flush=True)

    # registration.Update()
    # print(registration.GetOptimizer().GetStopConditionDescription())

    # finalParameters = outputBSplineTransform.GetParameters()
    # print(finalParameters)