import itk
# print(itk.Version.GetITKVersion())

# Load the fixed and moving images
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