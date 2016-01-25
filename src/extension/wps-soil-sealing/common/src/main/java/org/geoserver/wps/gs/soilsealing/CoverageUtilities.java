/* Copyright (c) 2001 - 2014 OpenPlans - www.openplans.org. All rights 
 * reserved. This code is licensed under the GPL 2.0 license, available at the 
 * root application directory.
 */
package org.geoserver.wps.gs.soilsealing;

import java.awt.Shape;
import java.awt.geom.AffineTransform;
import java.awt.geom.NoninvertibleTransformException;
import java.awt.image.DataBuffer;
import java.io.File;
import java.util.Collection;

import javax.media.jai.JAI;
import javax.media.jai.ROI;
import javax.media.jai.ROIShape;

import org.geotools.coverage.GridSampleDimension;
import org.geotools.coverage.TypeMap;
import org.geotools.coverage.grid.GridCoverage2D;
import org.geotools.coverage.grid.io.AbstractGridCoverage2DReader;
import org.geotools.data.DataSourceException;
import org.geotools.feature.simple.SimpleFeatureTypeBuilder;
import org.geotools.geometry.jts.JTS;
import org.geotools.geometry.jts.LiteShape2;
import org.geotools.process.ProcessException;
import org.geotools.referencing.operation.transform.AffineTransform2D;
import org.geotools.referencing.operation.transform.ProjectiveTransform;
import org.geotools.util.Utilities;
import org.geotools.utils.imageoverviews.OverviewsEmbedder;
import org.geotools.utils.progress.ExceptionEvent;
import org.geotools.utils.progress.ProcessingEvent;
import org.geotools.utils.progress.ProcessingEventListener;
import org.jaitools.imageutils.ROIGeometry;
import org.opengis.coverage.SampleDimensionType;
import org.opengis.coverage.grid.GridEnvelope;
import org.opengis.feature.simple.SimpleFeatureType;
import org.opengis.geometry.MismatchedDimensionException;
import org.opengis.parameter.GeneralParameterValue;
import org.opengis.parameter.ParameterDescriptor;
import org.opengis.parameter.ParameterValue;
import org.opengis.referencing.operation.TransformException;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.MultiPolygon;
import com.vividsolutions.jts.geom.Point;
import com.vividsolutions.jts.geom.Polygon;
import com.vividsolutions.jts.simplify.DouglasPeuckerSimplifier;

/**
 * A set of utilities methods for the Grid Coverage package. Those methods are not really rigorous; must of them should be seen as temporary
 * implementations.
 * 
 * @author Simone Giannecchini, GeoSolutions
 * @author Alessio Fabiani, GeoSolutions
 */
public class CoverageUtilities {
    /**
     * Do not allows instantiation of this class.
     */
    private CoverageUtilities() {
    }

    /**
     * Creates a {@link SimpleFeatureType} that exposes a coverage as a collections of feature points, mapping the centre of each pixel as a point
     * plus all the bands as attributes.
     * 
     * <p>
     * The FID is the long that combines x+y*width.
     * 
     * @param gc2d the {@link GridCoverage2D} to wrap.
     * @param geometryClass the class for the geometry.
     * @return a {@link SimpleFeatureType} or <code>null</code> in case we are unable to wrap the coverage
     */
    public static SimpleFeatureType createFeatureType(final GridCoverage2D gc2d,
            final Class<? extends Geometry> geometryClass) {

        // checks
        Utilities.ensureNonNull("gc2d", gc2d);

        // building a feature type for this coverage
        final SimpleFeatureTypeBuilder ftBuilder = new SimpleFeatureTypeBuilder();
        ftBuilder.setName(gc2d.getName().toString());
        ftBuilder.setNamespaceURI("http://www.geotools.org/");

        // CRS
        ftBuilder.setCRS(gc2d.getCoordinateReferenceSystem2D());
        // ftBuilder.setCRS(DefaultEngineeringCRS.GENERIC_2D);

        // TYPE is as follows the_geom | band
        ftBuilder.setDefaultGeometry("the_geom");
        ftBuilder.add("the_geom", geometryClass);
        if (!geometryClass.equals(Point.class)) {
            ftBuilder.add("value", Double.class);
        } else {

            // get sample type on bands
            final GridSampleDimension[] sampleDimensions = gc2d.getSampleDimensions();
            for (GridSampleDimension sd : sampleDimensions) {
                final SampleDimensionType sdType = sd.getSampleDimensionType();
                final int dataBuffType = TypeMap.getDataBufferType(sdType);

                // TODO I think this should be a public utility inside the FeatureUtilities class
                @SuppressWarnings("rawtypes")
                final Class bandClass;
                switch (dataBuffType) {
                case DataBuffer.TYPE_BYTE:
                    bandClass = Byte.class;
                    break;
                case DataBuffer.TYPE_DOUBLE:
                    bandClass = Double.class;
                    break;
                case DataBuffer.TYPE_FLOAT:
                    bandClass = Float.class;
                    break;
                case DataBuffer.TYPE_INT:
                    bandClass = Integer.class;
                    break;
                case DataBuffer.TYPE_SHORT:
                case DataBuffer.TYPE_USHORT:
                    bandClass = Short.class;
                    break;
                case DataBuffer.TYPE_UNDEFINED:
                default:
                    return null;
                }
                ftBuilder.add(sd.getDescription().toString(), bandClass);

            }

        }
        return ftBuilder.buildFeatureType();
    }

    // public static RangeLookupTable getRangeLookupTable(
    // final List<Range> classificationRanges,
    // final Number noDataValue) {
    //
    // return getRangeLookupTable(classificationRanges, noDataValue, noDataValue.getClass());
    // }

    // public static RangeLookupTable getRangeLookupTable(
    // final List<Range> classificationRanges,
    // final Number noDataValue,
    // final Class clazz) {
    // return getRangeLookupTable(classificationRanges, null, noDataValue, noDataValue.getClass());
    // }

    // public static RangeLookupTable getRangeLookupTable(
    // final List<Range> classificationRanges,
    // final int[] outputPixelValues,
    // final Number noDataValue) {
    // return getRangeLookupTable(classificationRanges, outputPixelValues, noDataValue, noDataValue.getClass());
    // }

    // public static RangeLookupTable getRangeLookupTable(
    // final List<Range> classificationRanges,
    // final int[] outputPixelValues,
    // final Number noDataValue,
    // final Class clazz) {
    //
    // final RangeLookupTable rlt = new RangeLookupTable(noDataValue);
    // final int size= classificationRanges.size();
    // final boolean useCustomOutputPixelValues = outputPixelValues != null && outputPixelValues.length == size;
    // for (int i = 0; i < size; i++) {
    // final int reference = useCustomOutputPixelValues ? outputPixelValues [i] : i + 1;
    // rlt.add(classificationRanges.get(i), convert(reference, noDataValue.getClass()));
    // }
    // return rlt;
    // }

    // @SuppressWarnings("unchecked")
    // public static <T extends Number & Comparable> T guessNoDataValue(Class<T> type){
    // if (type == null) {
    // return null;
    // } else if (Double.class.equals(type)) {
    // return (T) new Double(Double.NaN);
    // } else if (Float.class.equals(type)) {
    // return (T) new Float(Float.NaN);
    // } else if (Integer.class.equals(type)) {
    // return (T) Integer.valueOf(Integer.MIN_VALUE);
    // } else if (Byte.class.equals(type)) {
    // return (T) Byte.valueOf(0));
    // } else if (Short.class.equals(type)) {
    // return (T) Byte.valueOf(0));
    // } else {
    // throw new UnsupportedOperationException("Class " + type
    // + " can't be used in a value Range");
    // }
    // }

    public static Number convert(Number val, Class<? extends Number> type) {
        if (val == null) {
            return null;
        } else if (Double.class.equals(type)) {
            if (val instanceof Double) {
                return val;
            }
            return Double.valueOf(val.doubleValue());
        } else if (Float.class.equals(type)) {
            if (val instanceof Float) {
                return val;
            }
            return Float.valueOf(val.floatValue());
        } else if (Integer.class.equals(type)) {
            if (val instanceof Integer) {
                return val;
            }
            return Integer.valueOf(val.intValue());
        } else if (Byte.class.equals(type)) {
            if (val instanceof Byte) {
                return val;
            }
            return Byte.valueOf(val.byteValue());
        } else if (Short.class.equals(type)) {
            if (val instanceof Short) {
                return val;
            }
            return Short.valueOf(val.shortValue());
        } else {
            throw new UnsupportedOperationException("Class " + type
                    + " can't be used in a value Range");
        }
    }

    // public static <T extends Number & Comparable> Range<T> convertRange(Range src, Class<T> type) {
    // return new Range<T>(convert(src.getMin(), type), src.isMinIncluded(), convert(src.getMax(),
    // type), src.isMaxIncluded());
    // }

    // public static <T extends Number & Comparable<T>> Class<T> mapDataBufferType(int type) {
    //
    // switch (type) {
    // case DataBuffer.TYPE_BYTE:
    // return (Class<T>)Byte.class;
    // case DataBuffer.TYPE_USHORT:
    // return (Class<T>)Short.class;
    // case DataBuffer.TYPE_SHORT:
    // return (Class<T>)Short.class;
    // case DataBuffer.TYPE_INT:
    // return (Class<T>)Integer.class;
    // case DataBuffer.TYPE_FLOAT:
    // return (Class<T>)Float.class;
    // case DataBuffer.TYPE_DOUBLE:
    // return (Class<T>) Double.class;
    // default:
    // throw new IllegalArgumentException("Unknown DataBuffer type " + type);
    // }
    // }

    /**
     * Replace or add the provided parameter in the read parameters
     */
    public static <T> GeneralParameterValue[] replaceParameter(
            GeneralParameterValue[] readParameters, Object value, ParameterDescriptor<T> pd) {

        // scan all the params looking for the one we want to add
        for (GeneralParameterValue gpv : readParameters) {
            // in case of match of any alias add a param value to the lot
            if (gpv.getDescriptor().getName().equals(pd.getName())) {
                ((ParameterValue) gpv).setValue(value);
                // leave
                return readParameters;
            }
        }

        // add it to the array
        // add to the list
        GeneralParameterValue[] readParametersClone = new GeneralParameterValue[readParameters.length + 1];
        System.arraycopy(readParameters, 0, readParametersClone, 0, readParameters.length);
        final ParameterValue<T> pv = pd.createValue();
        pv.setValue(value);
        readParametersClone[readParameters.length] = pv;
        readParameters = readParametersClone;
        return readParameters;
    }

    /**
     * Utility method for transforming a geometry ROI into the raster space, using the provided affine transformation.
     * 
     * @param roi a {@link Geometry} in model space.
     * @param mt2d an {@link AffineTransform} that maps from raster to model space. This is already referred to the pixel corner.
     * @return a {@link ROI} suitable for using with JAI.
     * @throws ProcessException in case there are problems with ivnerting the provided {@link AffineTransform}. Very unlikely to happen.
     */
    public static ROI prepareROI(Geometry roi, AffineTransform mt2d) throws ProcessException {
        // transform the geometry to raster space so that we can use it as a ROI source
        Geometry rasterSpaceGeometry;
        try {
            rasterSpaceGeometry = JTS.transform(roi, new AffineTransform2D(mt2d.createInverse()));
        } catch (MismatchedDimensionException e) {
            throw new ProcessException(e);
        } catch (TransformException e) {
            throw new ProcessException(e);
        } catch (NoninvertibleTransformException e) {
            throw new ProcessException(e);
        }
        // System.out.println(rasterSpaceGeometry);
        // System.out.println(rasterSpaceGeometry.getEnvelopeInternal());

        // simplify the geometry so that it's as precise as the coverage, excess coordinates
        // just make it slower to determine the point in polygon relationship
        Geometry simplifiedGeometry = DouglasPeuckerSimplifier.simplify(rasterSpaceGeometry, 1);

        // build a shape using a fast point in polygon wrapper
        return new ROIGeometry(simplifiedGeometry);
    }

    /**
     * Transform the provided {@link Geometry} in world coordinates into
     * 
     * @param roi
     * @param gridToWorld
     * @return
     * @throws Exception
     */
    public static ROI prepareROI2(Geometry roi, AffineTransform gridToWorld) throws Exception {
        final Shape cropRoiLS2 = new LiteShape2(roi, ProjectiveTransform.create(gridToWorld)
                .inverse(), null, true, 1);
        return new ROIShape(cropRoiLS2);
    }

    /**
     * Transform the provided {@link Geometry} in world coordinates into
     * 
     * @param roi
     * @param gridToWorld
     * @return
     * @throws Exception
     */
    public static ROI prepareROIGeometry(Geometry roi, AffineTransform gridToWorld)
            throws Exception {

        Geometry projected = JTS.transform(roi, ProjectiveTransform.create(gridToWorld).inverse());

        return new ROIGeometry(projected);
    }

    /**
     * @param geomPolys
     * @param geom
     */
    public static void extractPolygons(Collection<Polygon> geomPolys, Geometry geom) {
        if (geom instanceof MultiPolygon) {
            for (int i = 0; i < ((MultiPolygon) geom).getNumGeometries(); i++) {
                Geometry g = ((MultiPolygon) geom).getGeometryN(i);
                if (g instanceof Polygon) {
                    if (g.getGeometryType().compareToIgnoreCase("Polygon") == 0) {
                        g.setSRID(geom.getSRID());
                        geomPolys.add((Polygon) g);
                    }
                }
            }
        } else if (geom instanceof Polygon) {
            if (geom.getGeometryType().compareToIgnoreCase("Polygon") == 0) {
                geomPolys.add((Polygon) geom);
            }
        }
    }

    /**
     * @param retValue
     * @return the number of steps processed, or 0 if none was done, or -1 on error.
     * @throws DataSourceException
     */
    public static int generateOverviews(AbstractGridCoverage2DReader abstractGridCoverage2DReader)
            throws DataSourceException {
        final File geotiffFile = (File) abstractGridCoverage2DReader.getSource();
        // ////
        // Adding Overviews
        // ////

        int tileH = 512;
        int tileW = 512;

        /** computing the number of steps **/
        GridEnvelope gridRange = abstractGridCoverage2DReader.getOriginalGridRange();

        int height = gridRange.getSpan(1);
        int width = gridRange.getSpan(0);

        int ratioH = (int) Math.ceil((1.0 * height) / tileH);
        int ratioW = (int) Math.ceil((1.0 * width) / tileW);

        int nStepsH = 0;
        int nStepsW = 0;

        if (ratioH >= 2) {
            nStepsH = (int) Math.floor(Math.log(ratioH) / Math.log(2));
        }

        if (ratioW >= 2) {
            nStepsW = (int) Math.floor(Math.log(ratioW) / Math.log(2));
        }

        int numSteps = Math.min(nStepsH, nStepsW);
        int downSampleSteps = 2;

        if (numSteps > 0) {
            final OverviewsEmbedder oe = new OverviewsEmbedder();
            oe.setDownsampleStep(downSampleSteps);
            oe.setNumSteps(numSteps);
            oe.setScaleAlgorithm(OverviewsEmbedder.SubsampleAlgorithm.Nearest.toString());
            oe.setTileCache(JAI.getDefaultInstance().getTileCache());
            // oe.setTileHeight(tileH);
            // oe.setTileWidth(tileW);
            oe.setSourcePath(geotiffFile.getAbsolutePath());

            EmbedderListener listener = new EmbedderListener(geotiffFile.getAbsolutePath());
            // add logger/listener
            oe.addProcessingEventListener(listener);

            // run
            oe.run(); // should block until terminated
            return listener.isSuccess() ? numSteps : -1;
        } else
            return 0;
    }

    static class EmbedderListener implements ProcessingEventListener {
        final String filename;

        boolean success = false;

        public EmbedderListener(String filename) {
            this.filename = filename;
        }

        public boolean isSuccess() {
            return success;
        }

        public void exceptionOccurred(ExceptionEvent event) {
            success = false;
        }

        public void getNotification(ProcessingEvent event) {
            if (event.getPercentage() == 100.0) {
                success = true;
            }
        }
    }
}
