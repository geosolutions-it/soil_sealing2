/**
 * 
 */
package org.geoserver.wfs.gs;

import java.util.Set;

import net.opengis.wfs.DeleteElementType;

import org.geoserver.catalog.Catalog;
import org.geoserver.catalog.CoverageStoreInfo;
import org.geoserver.catalog.LayerInfo;
import org.geoserver.catalog.ResourceInfo;
import org.geoserver.catalog.StoreInfo;
import org.geoserver.catalog.StyleInfo;
import org.geoserver.catalog.WorkspaceInfo;
import org.geoserver.wfs.TransactionEvent;
import org.geoserver.wfs.TransactionEventType;
import org.geoserver.wfs.TransactionListener;
import org.geoserver.wfs.WFSException;
import org.geotools.data.simple.SimpleFeatureCollection;
import org.geotools.data.simple.SimpleFeatureIterator;
import org.opengis.feature.Property;
import org.opengis.feature.simple.SimpleFeature;

/**
 * @author geosolutions
 *
 */
public class SoilSealingTransactionListener implements TransactionListener {

	private Catalog catalog;

	public SoilSealingTransactionListener(Catalog catalog) {
		this.catalog = catalog;
	}
	
	@SuppressWarnings("unused")
	@Override
	public void dataStoreChange(TransactionEvent event) throws WFSException {
		// TODO Auto-generated method stub
		TransactionEventType eventType = event.getType();
		
		if (eventType == TransactionEventType.PRE_DELETE) {
			Object eventSource = event.getSource();
			if (eventSource instanceof DeleteElementType) {
				DeleteElementType element = (DeleteElementType)eventSource;
				
				if ("changematrix".equals(element.getTypeName().getLocalPart()) ||
						"soilsealing".equals(element.getTypeName().getLocalPart())) {
					SimpleFeatureCollection features = event.getAffectedFeatures();
					SimpleFeatureIterator iterator = features.features();
					while (iterator.hasNext()) {
						SimpleFeature ft = iterator.next();
						Property workSpaceName = ft.getProperty("wsName");
						Property layerName = ft.getProperty("layerName");
						
						try {
							if (workSpaceName != null && layerName != null) {
								WorkspaceInfo ws = this.catalog.getWorkspaceByName((String) workSpaceName.getValue());
								
								CoverageStoreInfo store = this.catalog.getCoverageStoreByName(ws, (String)layerName.getValue());
								
								if (store != null) {
									// TODO...
								}
							}
							
							LayerInfo layer = this.catalog.getLayerByName((String)layerName.getValue());
							
							if (layer != null) {
								Set<StyleInfo> styles = layer.getStyles();
								ResourceInfo resource = layer.getResource();
								StoreInfo store = resource.getStore();
								
								// TODO...
							}
						} catch (Exception e) {
							//
						}
					}
				}
			}
		}
	}

}
