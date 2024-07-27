import itk

class CommandIterationUpdate(itk.Command):
    def __init__(self):
        self.optimizer = None
        super(CommandIterationUpdate, self).__init__()

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def execute(self, caller, event):
        if not isinstance(caller, itk.RegularStepGradientDescentOptimizer):
            return
        optimizer = caller
        if event.GetEventName() == 'IterationEvent':
            print(f"{optimizer.GetCurrentIteration()}   {optimizer.GetValue()}", flush=True)

def main():
    # print(f"Usage: fixedImageFile movingImageFile outputImagefile "
    #         "[differenceOutputfile] [differenceBeforeRegistration] [filenameForFinalTransformParameters] "
    #         "[useExplicitPDFderivatives] [useCachingBSplineWeights] [deformationField] "
    #         "[numberOfGridNodesInsideImageInOneDimensionCoarse] [numberOfGridNodesInsideImageInOneDimensionFine] "
    #         "[maximumStepLength] [maximumNumberOfIterations]")

    ImageDimension = 3
    PixelType = itk.UC  # 8-bit unsigned char

    FixedImageType = itk.Image[PixelType, ImageDimension]
    MovingImageType = itk.Image[PixelType, ImageDimension]

    SpaceDimension = ImageDimension
    SplineOrder = 3
    CoordinateRepType = itk.D

    RigidTransformType = itk.VersorRigid3DTransform[CoordinateRepType]
    AffineTransformType = itk.AffineTransform[CoordinateRepType, SpaceDimension]
    DeformableTransformType = itk.BSplineTransform[CoordinateRepType, SpaceDimension, SplineOrder]

    TransformInitializerType = itk.CenteredTransformInitializer[RigidTransformType, FixedImageType, MovingImageType]
    OptimizerType = itk.RegularStepGradientDescentOptimizer
    MetricType = itk.MattesMutualInformationImageToImageMetric[FixedImageType, MovingImageType]
    InterpolatorType = itk.LinearInterpolateImageFunction[MovingImageType, itk.D]

    RegistrationType = itk.ImageRegistrationMethod[FixedImageType, MovingImageType]

    metric = MetricType.New()
    optimizer = OptimizerType.New()
    interpolator = InterpolatorType.New()
    registration = RegistrationType.New()

    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    registration.SetInterpolator(interpolator)

    IdentityTransformType = itk.IdentityTransform[CoordinateRepType, SpaceDimension]
    identityTransform = IdentityTransformType.New()

    FixedImageReaderType = itk.ImageFileReader[FixedImageType]
    MovingImageReaderType = itk.ImageFileReader[MovingImageType]

    fixedImageReader = FixedImageReaderType.New()
    movingImageReader = MovingImageReaderType.New()

    fixedImageReader.SetFileName("./Data/case6_gre1.nrrd")
    movingImageReader.SetFileName("./Data/case6_gre2.nrrd")

    try:
        fixedImageReader.Update()
        movingImageReader.Update()
    except itk.ITKException as e:
        print(f"Exception caught: {e}", flush=True)
        return itk.ExitFailure

    fixedImage = fixedImageReader.GetOutput()

    registration.SetFixedImage(fixedImage)
    registration.SetMovingImage(movingImageReader.GetOutput())

    # chronometer = itk.TimeProbesCollectorBase.New()
    # memorymeter = itk.MemoryProbesCollectorBase.New()

    metric.SetNumberOfHistogramBins(50)
    fixedRegion = fixedImage.GetBufferedRegion()
    numberOfPixels = fixedRegion.GetNumberOfPixels()
    metric.ReinitializeSeed(76926294)

    # metric.SetUseExplicitPDFDerivatives(bool(int(sys.argv[7])))

    # metric.SetUseCachingOfBSplineWeights(bool(int(sys.argv[8])))

    initializer = TransformInitializerType.New()
    rigidTransform = RigidTransformType.New()
    initializer.SetTransform(rigidTransform)
    initializer.SetFixedImage(fixedImageReader.GetOutput())
    initializer.SetMovingImage(movingImageReader.GetOutput())
    initializer.MomentsOn()

    print("Starting Rigid Transform Initialization", flush=True)
    # memorymeter.Start("Rigid Initialization")
    # chronometer.Start("Rigid Initialization")

    initializer.InitializeTransform()

    # chronometer.Stop("Rigid Initialization")
    # memorymeter.Stop("Rigid Initialization")

    print("Rigid Transform Initialization completed", flush=True)

    registration.SetFixedImageRegion(fixedRegion)
    registration.SetInitialTransformParameters(rigidTransform.GetParameters())
    registration.SetTransform(rigidTransform)

    OptimizerScalesType = itk.Vector[itk.D, SpaceDimension]
    optimizerScales = OptimizerScalesType()
    optimizerScales.Fill(1.0)
    translationScale = 1.0 / 1000.0

    optimizerScales[3:6] = [translationScale] * 3
    optimizer.SetScales(optimizerScales)
    optimizer.SetMaximumStepLength(0.2000)
    optimizer.SetMinimumStepLength(0.0001)
    optimizer.SetNumberOfIterations(200)
    metric.SetNumberOfSpatialSamples(10000)

    observer = CommandIterationUpdate.New()
    observer.set_optimizer(optimizer)
    optimizer.AddObserver(itk.IterationEvent(), observer)

    print("Starting Rigid Registration", flush=True)

    try:
        # memorymeter.Start("Rigid Registration")
        # chronometer.Start("Rigid Registration")
        registration.Update()
        # chronometer.Stop("Rigid Registration")
        # memorymeter.Stop("Rigid Registration")
        print(f"Optimizer stop condition = {registration.GetOptimizer().GetStopConditionDescription()}", flush=True)
    except itk.ITKException as e:
        print(f"Exception caught: {e}", flush=True)
        return itk.ExitFailure

    print("Rigid Registration completed", flush=True)

    rigidTransform.SetParameters(registration.GetLastTransformParameters())

    affineTransform = AffineTransformType.New()
    affineTransform.SetCenter(rigidTransform.GetCenter())
    affineTransform.SetTranslation(rigidTransform.GetTranslation())
    affineTransform.SetMatrix(rigidTransform.GetMatrix())

    registration.SetTransform(affineTransform)
    registration.SetInitialTransformParameters(affineTransform.GetParameters())

    optimizerScales = itk.Vector[float, affineTransform.GetNumberOfParameters()]
    optimizerScales.Fill(1.0)
    optimizerScales[9:12] = [translationScale] * 3
    optimizer.SetScales(optimizerScales)
    optimizer.SetMaximumStepLength(0.2000)
    optimizer.SetMinimumStepLength(0.0001)
    optimizer.SetNumberOfIterations(200)
    metric.SetNumberOfSpatialSamples(50000)

    print("Starting Affine Registration", flush=True)

    try:
        # memorymeter.Start("Affine Registration")
        # chronometer.Start("Affine Registration")
        registration.Update()
        # chronometer.Stop("Affine Registration")
        # memorymeter.Stop("Affine Registration")
    except itk.ITKException as e:
        print(f"Exception caught: {e}", flush=True)
        return itk.ExitFailure

    print("Affine Registration completed", flush=True)

    affineTransform.SetParameters(registration.GetLastTransformParameters())

    bsplineTransformCoarse = DeformableTransformType.New()
    numberOfGridNodesInOneDimensionCoarse = 5
    fixedPhysicalDimensions = [0] * SpaceDimension
    meshSize = [numberOfGridNodesInOneDimensionCoarse - SplineOrder] * SpaceDimension
    fixedOrigin = [0] * SpaceDimension

    for i in range(SpaceDimension):
        fixedOrigin[i] = fixedImage.GetOrigin()[i]
        fixedPhysicalDimensions[i] = fixedImage.GetSpacing()[i] * (fixedImage.GetLargestPossibleRegion().GetSize()[i] - 1)

    bsplineTransformCoarse.SetTransformDomainOrigin(fixedOrigin)
    bsplineTransformCoarse.SetTransformDomainPhysicalDimensions(fixedPhysicalDimensions)
    bsplineTransformCoarse.SetTransformDomainMeshSize(meshSize)
    bsplineTransformCoarse.SetTransformDomainDirection(fixedImage.GetDirection())

    numberOfBSplineParameters = bsplineTransformCoarse.GetNumberOfParameters()
    optimizerScales = itk.Vector[float, numberOfBSplineParameters]
    optimizerScales.Fill(1.0)
    optimizer.SetScales(optimizerScales)

    initialDeformableTransformParameters = itk.Vector[float, numberOfBSplineParameters]
    initialDeformableTransformParameters.Fill(0.0)

    CompositeTransformType = itk.CompositeTransform[CoordinateRepType, SpaceDimension]
    compositeTransform = CompositeTransformType.New()
    compositeTransform.AddTransform(affineTransform)
    compositeTransform.AddTransform(bsplineTransformCoarse)
    compositeTransform.SetOnlyMostRecentTransformToOptimizeOn()

    bsplineTransformCoarse.SetParameters(initialDeformableTransformParameters)
    registration.SetInitialTransformParameters(bsplineTransformCoarse.GetParameters())
    registration.SetTransform(compositeTransform)

    optimizer.SetMaximumStepLength(10.0)
    optimizer.SetMinimumStepLength(0.01)
    optimizer.SetRelaxationFactor(0.7)
    optimizer.SetNumberOfIterations(50)

    metric.SetNumberOfSpatialSamples(numberOfBSplineParameters * 100)

    print("Starting Deformable Registration Coarse Grid", flush=True)

    try:
        # memorymeter.Start("Deformable Registration Coarse")
        # chronometer.Start("Deformable Registration Coarse")
        registration.Update()
        # chronometer.Stop("Deformable Registration Coarse")
        # memorymeter.Stop("Deformable Registration Coarse")
    except itk.ITKException as e:
        print(f"Exception caught: {e}", flush=True)
        return itk.ExitFailure

    print("Deformable Registration Coarse Grid completed", flush=True)

    bsplineTransformFine = DeformableTransformType.New()
    numberOfGridNodesInOneDimensionFine = numberOfGridNodesInOneDimensionCoarse * 2
    meshSize = [numberOfGridNodesInOneDimensionFine - SplineOrder] * SpaceDimension

    bsplineTransformFine.SetTransformDomainOrigin(fixedOrigin)
    bsplineTransformFine.SetTransformDomainPhysicalDimensions(fixedPhysicalDimensions)
    bsplineTransformFine.SetTransformDomainMeshSize(meshSize)
    bsplineTransformFine.SetTransformDomainDirection(fixedImage.GetDirection())

    initialDeformableTransformParameters = itk.Vector[float, bsplineTransformFine.GetNumberOfParameters()]
    initialDeformableTransformParameters.Fill(0.0)

    bsplineTransformFine.SetParameters(initialDeformableTransformParameters)
    compositeTransform.RemoveTransform(bsplineTransformCoarse)
    compositeTransform.AddTransform(bsplineTransformFine)
    compositeTransform.SetOnlyMostRecentTransformToOptimizeOn()

    optimizer.SetMaximumStepLength(10.0)
    optimizer.SetMinimumStepLength(0.01)
    optimizer.SetRelaxationFactor(0.7)
    optimizer.SetNumberOfIterations(200)
    metric.SetNumberOfSpatialSamples(numberOfBSplineParameters * 100)

    print("Starting Deformable Registration Fine Grid", flush=True)

    try:
        # memorymeter.Start("Deformable Registration Fine")
        # chronometer.Start("Deformable Registration Fine")
        registration.Update()
        # chronometer.Stop("Deformable Registration Fine")
        # memorymeter.Stop("Deformable Registration Fine")
    except itk.ITKException as e:
        print(f"Exception caught: {e}", flush=True)
        return itk.ExitFailure

    print("Deformable Registration Fine Grid completed", flush=True)

    finalParameters = registration.GetLastTransformParameters()
 
    bsplineTransformFine.SetParameters(finalParameters)
    
    ResampleFilterType = itk.ResampleImageFilter[MovingImageType, FixedImageType]
    resample = ResampleFilterType.New()
    
    resample.SetTransform(bsplineTransformFine)
    resample.SetInput(movingImageReader.GetOutput())
    
    resample.SetSize(fixedImage.GetLargestPossibleRegion().GetSize())
    resample.SetOutputOrigin(fixedImage.GetOrigin())
    resample.SetOutputSpacing(fixedImage.GetSpacing())
    resample.SetOutputDirection(fixedImage.GetDirection())
    
    resample.SetDefaultPixelValue(0) # 100 or 128
    
    OutputPixelType = itk.US # was short
    OutputImageType = itk.Image[OutputPixelType, ImageDimension]
    
    CastFilterType = itk.CastImageFilter[FixedImageType, OutputImageType]
    WriterType = itk.ImageFileWriter[OutputImageType]
    
    writer = WriterType.New()
    caster = CastFilterType.New()
    
    writer.SetFileName("./case6_gre2_registered.nrrd")
    
    caster.SetInput(resample.GetOutput())
    writer.SetInput(caster.GetOutput())
    
    print("Writing resampled moving image...", flush=True)
    
    try:
        writer.Update()
    except itk.ExceptionObject as err:
        print(err, flush=True)
        return 1
    
    print("Done!")

if __name__ == "__main__":
    main()