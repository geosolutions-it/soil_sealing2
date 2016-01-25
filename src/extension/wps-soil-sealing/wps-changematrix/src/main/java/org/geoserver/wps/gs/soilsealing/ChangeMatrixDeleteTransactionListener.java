/* Copyright (c) 2001 - 2014 OpenPlans - www.openplans.org. All rights 
 * reserved. This code is licensed under the GPL 2.0 license, available at the 
 * root application directory.
 */
package org.geoserver.wps.gs.soilsealing;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.geoserver.catalog.Catalog;
import org.geoserver.catalog.CoverageInfo;
import org.geoserver.catalog.LayerInfo;
import org.geoserver.catalog.ResourceInfo;
import org.geoserver.catalog.StoreInfo;
import org.geoserver.catalog.WorkspaceInfo;
import org.geoserver.config.GeoServer;
import org.geoserver.wfs.TransactionEvent;
import org.geoserver.wfs.TransactionEventType;
import org.geoserver.wfs.TransactionListener;
import org.geoserver.wfs.WFSException;
import org.geotools.data.DataUtilities;
import org.geotools.util.logging.Logging;
import org.opengis.feature.simple.SimpleFeature;

public class ChangeMatrixDeleteTransactionListener implements TransactionListener {

    protected static final Logger LOGGER = Logging
            .getLogger(ChangeMatrixDeleteTransactionListener.class);

    protected GeoServer geoServer;

    protected Catalog catalog;

    public ChangeMatrixDeleteTransactionListener(GeoServer geoServer) {
        this.geoServer = geoServer;
        this.catalog = geoServer.getCatalog();
    }

    public void clear() {
    }

    public void dataStoreChange(TransactionEvent event) throws WFSException {
        String typeName = event.getAffectedFeatures().getSchema().getTypeName();

        if (LOGGER.isLoggable(Level.FINE)) {
            LOGGER.fine("ChangeMatrixDeleteTransactionListener - catched request for FeatureType: "
                    + typeName);
        }

        // check the correct event type and the correct FeatureType Name
        if (typeName.equalsIgnoreCase("changematrix") && TransactionEventType.PRE_DELETE == event.getType()) {
            List features = new ArrayList();
            features.addAll(DataUtilities.list(event.getAffectedFeatures()));
            for (Object ft : features) {
                if (ft instanceof SimpleFeature) {
                    WorkspaceInfo ws = null;
                    StoreInfo storeInfo = null;
                    try {
                        // retrieve workspace and store as inserted in the feature attributes
                        SimpleFeature next = (SimpleFeature) ft;

                        if (LOGGER.isLoggable(Level.FINE)) {
                            LOGGER.fine("ChangeMatrixDeleteTransactionListener - checking feature: "
                                    + next.getID());
                        }

                        String wsName = (String) next.getAttribute("wsName");
                        String itemStatus = (String) next.getAttribute("itemStatus");
                        String layerName = (String) next.getAttribute("layerName");

                        if (LOGGER.isLoggable(Level.FINE)) {
                            LOGGER.fine("ChangeMatrixDeleteTransactionListener - wsName: " + wsName);
                            LOGGER.fine("ChangeMatrixDeleteTransactionListener - layerName: "
                                    + layerName);
                            LOGGER.fine("ChangeMatrixDeleteTransactionListener - itemStatus: "
                                    + itemStatus);
                        }

                        if (itemStatus != null && itemStatus.equals("RUNNING")) {
                            LOGGER.severe("Exception occurred during Deletion: Could not remove a feature in status 'RUNNING'");
                            // throw new WFSException("Exception occurred during Deletion: Could not remove a feature in status 'RUNNING'");
                            continue;
                        }

                        if (wsName != null && itemStatus != null && itemStatus.equals("COMPLETED")
                                && wsName.length() > 0) {
                            // being sure the workspace exists in the catalog
                            ws = catalog.getWorkspaceByName(wsName);
                            if (ws == null) {
                                LOGGER.severe("Could not retrive WorkSpace " + wsName
                                        + " into the Catalog");
                            }

                            if (ws != null) {
                                // being sure the store exists in the catalog
                                CoverageInfo coverageInfo = catalog.getCoverageByName(wsName,
                                        layerName);
                                storeInfo = catalog.getDataStoreByName(layerName.trim()) != null ? catalog
                                        .getDataStoreByName(layerName.trim()) : catalog
                                        .getDataStoreByName(ws.getName(), layerName.trim());
                                storeInfo = (storeInfo == null ? (catalog
                                        .getCoverageStoreByName(layerName.trim()) != null ? catalog
                                        .getCoverageStoreByName(layerName.trim()) : catalog
                                        .getCoverageStoreByName(ws.getName(), layerName.trim()))
                                        : storeInfo);

                                if (storeInfo == null) {
                                    LOGGER.severe("Could not retrive Store " + layerName
                                            + " into the Catalog");
                                } else {
                                    // drill down into layers (into resources since we cannot scan layers)
                                    int layersInStore = 0;
                                    try {
                                        List<ResourceInfo> resources = catalog.getResourcesByStore(
                                                storeInfo, ResourceInfo.class);
                                        for (ResourceInfo ri : resources) {
                                            List<LayerInfo> layers = catalog.getLayers(ri);
                                            if (!layers.isEmpty()) {
                                                for (LayerInfo li : layers) {
                                                    // counting the store layers, if 0 we can remove the whole store too ...
                                                    layersInStore++;
                                                    // we need to check the layer name start, since for the coverages a timestamp is attached to the
                                                    // name

                                                    if (LOGGER.isLoggable(Level.FINE)) {
                                                        LOGGER.fine("ChangeMatrixDeleteTransactionListener - going to check layer: "
                                                                + li.getName()
                                                                + " against feture layer attribute: "
                                                                + layerName);
                                                    }

                                                    if (layerName != null && li != null
                                                            && li.getName() != null
                                                            && li.getName().equals(layerName)) {
                                                        catalog.remove(li);
                                                        catalog.remove(li.getResource());
                                                        layersInStore--;

                                                        if (LOGGER.isLoggable(Level.FINE)) {
                                                            LOGGER.fine("ChangeMatrixDeleteTransactionListener - removed layer: "
                                                                    + li.getName());
                                                        }
                                                    }
                                                }
                                            }

                                            // //
                                            // Going to remove the resource geotiff and its parent folder
                                            // //
                                            try {
                                                File storeResource = null;
                                                LOGGER.info("ChangeMatrixDeleteTransactionListener - ConnectionParameters URL: "
                                                        + ((CoverageInfo) ri).getStore().getURL());
                                                String urlTxt = ((CoverageInfo) ri).getStore()
                                                        .getURL();
                                                if (urlTxt.startsWith("file:")
                                                        && !urlTxt.startsWith("file:///")) {
                                                    urlTxt = urlTxt.substring("file:".length());
                                                    storeResource = new File(urlTxt);
                                                }

                                                // try to delete the file several times since it may be locked by catalog for some reason
                                                if (storeResource != null && storeResource.exists()
                                                        && storeResource.isFile()
                                                        && storeResource.canWrite()) {
                                                    final int retries = 10;
                                                    int tryDelete = 0;
                                                    for (; tryDelete < 5; tryDelete++) {
                                                        if (storeResource.delete()) {
                                                            if (LOGGER.isLoggable(Level.FINE)) {
                                                                LOGGER.fine("ChangeMatrixDeleteTransactionListener - deleted storeResource: "
                                                                        + storeResource
                                                                                .getAbsolutePath());
                                                            }

                                                            // removing also the parent folder if empty.
                                                            storeResource.getParentFile().delete();

                                                            break;
                                                        } else {
                                                            if (LOGGER.isLoggable(Level.FINE)) {
                                                                LOGGER.fine("ChangeMatrixDeleteTransactionListener - could not delete storeResource: "
                                                                        + storeResource
                                                                                .getAbsolutePath()
                                                                        + " ... wait 5 secs and retry...");
                                                            }

                                                            // wait 5 seconds and try again...
                                                            Thread.sleep(5000);
                                                        }
                                                    }

                                                    if (tryDelete > retries) {
                                                        LOGGER.severe("ChangeMatrixDeleteTransactionListener - Could not delete file "
                                                                + storeResource.getAbsolutePath()
                                                                + " from the FileSystem");
                                                    }
                                                }
                                            } catch (Exception e) {
                                                LOGGER.warning("ChangeMatrixDeleteTransactionListener - Could not cleanup store and layer resource from the FileSystem: "
                                                        + e.getLocalizedMessage());
                                            }
                                        }

                                        // the store does not contain layers anymore, lets remove it from the catalog then

                                        if (LOGGER.isLoggable(Level.FINE)) {
                                            LOGGER.fine("ChangeMatrixDeleteTransactionListener - remaining layers for this store: "
                                                    + layersInStore);
                                        }

                                        if (layersInStore == 0
                                                && storeInfo.getName().equals(layerName)) {
                                            // geoServer.reload();
                                            catalog.remove(storeInfo);

                                            if (LOGGER.isLoggable(Level.FINE)) {
                                                LOGGER.fine("ChangeMatrixDeleteTransactionListener - removed store: "
                                                        + storeInfo.getName());
                                            }
                                        }
                                    } catch (Exception e) {
                                        LOGGER.severe("ChangeMatrixDeleteTransactionListener - Could not remove store and layer from catalog: "
                                                + e.getLocalizedMessage());
                                    }
                                }
                            }
                        }

                        // //
                        // Going to remove the Octave source file used to generate the layer
                        // //
                        String srcPath = (String) next.getAttribute("srcPath");

                        if (LOGGER.isLoggable(Level.FINE)) {
                            LOGGER.fine("ChangeMatrixDeleteTransactionListener - going to delete file: "
                                    + srcPath);
                        }

                        if (ws != null && storeInfo != null && srcPath != null
                                && srcPath.length() > 0) {
                            File file = new File(srcPath);

                            // try to delete the file several times since it may be locked by catalog for some reason
                            if (file.exists() && file.isFile() && file.canWrite()) {
                                final int retries = 10;
                                int tryDelete = 0;
                                for (; tryDelete < 5; tryDelete++) {
                                    if (file.delete()) {
                                        if (LOGGER.isLoggable(Level.FINE)) {
                                            LOGGER.fine("ChangeMatrixDeleteTransactionListener - deleted file: "
                                                    + srcPath);
                                        }

                                        break;
                                    } else {
                                        if (LOGGER.isLoggable(Level.FINE)) {
                                            LOGGER.fine("ChangeMatrixDeleteTransactionListener - could not delete file: "
                                                    + srcPath + " ... wait 5 secs and retry...");
                                        }

                                        // wait 5 seconds and try again...
                                        Thread.sleep(5000);
                                    }
                                }

                                if (tryDelete > retries) {
                                    LOGGER.severe("ChangeMatrixDeleteTransactionListener - Could not delete file "
                                            + srcPath + " from the FileSystem");
                                }
                            }
                        }

                        // //
                        // Going to remove the Octave sound velocity profile file uploaded and used to generate the layer
                        // //
                        String soundVelocityProfile = (String) next
                                .getAttribute("soundVelocityProfile");

                        if (LOGGER.isLoggable(Level.FINE)) {
                            LOGGER.fine("ChangeMatrixDeleteTransactionListener - going to delete file: "
                                    + soundVelocityProfile);
                        }

                        if (soundVelocityProfile != null && soundVelocityProfile.length() > 0) {
                            File file = new File(soundVelocityProfile);

                            // try to delete the file several times since it may be locked by catalog for some reason
                            if (file.exists() && file.isFile() && file.canWrite()) {
                                final int retries = 10;
                                int tryDelete = 0;
                                for (; tryDelete < 5; tryDelete++) {
                                    if (file.delete()) {
                                        if (LOGGER.isLoggable(Level.FINE)) {
                                            LOGGER.fine("ChangeMatrixDeleteTransactionListener - deleted file: "
                                                    + soundVelocityProfile);
                                        }

                                        break;
                                    } else {
                                        if (LOGGER.isLoggable(Level.FINE)) {
                                            LOGGER.fine("ChangeMatrixDeleteTransactionListener - could not delete file: "
                                                    + soundVelocityProfile
                                                    + " ... wait 5 secs and retry...");
                                        }

                                        // wait 5 seconds and try again...
                                        Thread.sleep(5000);
                                    }
                                }

                                if (tryDelete > retries) {
                                    LOGGER.severe("ChangeMatrixDeleteTransactionListener - Could not delete file "
                                            + soundVelocityProfile + " from the FileSystem");
                                }
                            }
                        }
                    } catch (Exception e) {
                        LOGGER.severe("ChangeMatrixDeleteTransactionListener - Exception occurred during Deletion: "
                                + e.getLocalizedMessage());
                        throw new WFSException(e);
                    } finally {
                    }
                }
            }
        }
    }
}