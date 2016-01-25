package org.geoserver.wps.area;

import javax.media.jai.OperationDescriptorImpl;
import javax.media.jai.registry.RenderedRegistryMode;

import org.geotools.geometry.jts.ReferencedEnvelope;

/**
 * This class is an {@link OperationDescriptorImpl} for the Area operation. It describes the parameters to set for calling the "Area" operation.
 */
public class AreaDescriptor extends OperationDescriptorImpl {

    /** Default value for the Area Multiplier (Should be substituted by a value related to the input image classes ) */
    private static final double DEFAULT_MULTIPLIER = 1d;

    /** Serial Version UID associated to the class */
    private static final long serialVersionUID = -6996896157854316840L;

    /** Index associated to the Envelope input parameter */
    public static final int ENVELOPE_INDEX = 0;

    /** Index associated to the Multiplier input parameter */
    public static final int PIXEL_MULTY_INDEX = 1;

    /** Index associated to the validValues input parameter */
    public static final int VALID_VALUES_INDEX = 2;

    /** Index associated to the ROI input parameter */
    public static final int ROI_INDEX = 3;

    /** Names of all the input parameters */
    public static final String[] PARAM_NAMES = { "envelope", "multiplier", "validValues", "roi" };

    /** Classes of all the input parameters */
    private static final Class<?>[] PARAM_CLASSES = { ReferencedEnvelope.class, Double.class,
            java.util.Set.class, javax.media.jai.ROI.class };

    /** Default valuescfor all the input parameters */
    private static final Object[] PARAM_DEFAULTS = { NO_PARAMETER_DEFAULT, DEFAULT_MULTIPLIER,
            null, null };

    /** Constructor. */
    public AreaDescriptor() {
        super(new String[][] { { "GlobalName", "area" }, { "LocalName", "area" },
                { "Vendor", "org.soil.sealing" },
                { "Description", "Calculate change matrix between two images" },
                { "DocURL", "http://www.geotools.org" }, { "Version", "1.0.0" },
                { "arg0Desc", "Envelope of the source image" },
                { "arg1Desc", "Multiplier to multiply to the area value" },
                { "arg2Desc", "A Set of pixels which must be calculated" },
                { "arg3Desc", "Optional ROI to use" }

        },

        new String[] { RenderedRegistryMode.MODE_NAME }, // supported modes

                1, // number of sources

                PARAM_NAMES, PARAM_CLASSES, PARAM_DEFAULTS,

                null // valid values (none defined)
        );
    }
}
