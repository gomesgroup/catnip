import React, {useRef, useEffect} from 'react';

const MolecularViewer = ({moleculeData, format}) => {
    const viewerRef = useRef(null);

    useEffect(() => {
        if (viewerRef.current) {
            const viewer = window.$3Dmol.createViewer(viewerRef.current, {
                defaultcolors: window.$3Dmol.rasmolElementColors
            });

            viewer.addModel(moleculeData, "sdf");
            viewer.setStyle({}, {stick: {}});
            console.log(viewer)
            console.log(moleculeData)
            viewer.zoomTo(viewer.getModel());
            viewer.render();
        }
    }, [moleculeData, format]);

    return (
        <div ref={viewerRef} style={{width: '100%', height: '400px'}}></div>
    );
};

export default MolecularViewer;
