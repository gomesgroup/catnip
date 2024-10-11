import React from "react";
import "miew/dist/miew.min.css";
import {StandaloneStructServiceProvider} from "ketcher-standalone";
import {Editor} from "ketcher-react";
import {Ketcher} from "ketcher-core";
import "ketcher-react/dist/index.css";
import Miew from "miew";
import {ButtonsConfig} from "ketcher-react";

(window as any).Miew = Miew;


const structServiceProvider = new StandaloneStructServiceProvider();

const getHiddenButtonsConfig = (hiddenButtons: string[] | null = null): ButtonsConfig => {
    if (!hiddenButtons) return {};

    return hiddenButtons.reduce((acc: any, button) => {
        if (button) acc[button] = {hidden: true};

        return acc;
    }, {});
};

export class KetcherExample extends React.Component {
    ketcher: Ketcher;

    handleOnInit = async (ketcher: Ketcher) => {
        this.ketcher = ketcher;
        (window as any).ketcher = ketcher;
    };

    // This method will return the molecule sketch data
    getMoleculeData = () => {
        return this.ketcher ? this.ketcher.getSmiles() : '';
    }

    render() {
        return (
            <Editor
                errorHandler={(message: string) => null}
                staticResourcesUrl={""}
                structServiceProvider={structServiceProvider}
                onInit={this.handleOnInit}
                buttons={getHiddenButtonsConfig([
                    'arom',
                    'dearom',
                    'cip',
                    'check',
                    'analyse',
                    'recognize',
                    'help',
                    'about',
                    'settings',
                    'clean',
                    'layout',
                    'sgroup',
                    'sgroup-data',
                    'reaction-mapping-tools',
                    'reaction-automap',
                    'reaction-map',
                    'reaction-unmap',
                    'rgroup',
                    'rgroup-label',
                    'rgroup-fragment',
                    'rgroup-attpoints',
                    'text',
                    'enhanced-stereo',
                    'reaction-plus',
                    'arrows',
                    'reaction-arrow-open-angle',
                    'reaction-arrow-filled-triangle',
                    'reaction-arrow-filled-bow',
                    'reaction-arrow-dashed-open-angle',
                    'reaction-arrow-failed',
                    'reaction-arrow-both-ends-filled-triangle',
                    'reaction-arrow-equilibrium-filled-half-bow',
                    'reaction-arrow-equilibrium-filled-triangle',
                    'reaction-arrow-equilibrium-open-angle',
                    'reaction-arrow-unbalanced-equilibrium-filled-half-bow',
                    'reaction-arrow-unbalanced-equilibrium-open-half-angle',
                    'reaction-arrow-unbalanced-equilibrium-large-filled-half-bow',
                    'reaction-arrow-unbalanced-equilibrium-filled-half-triangle',
                    'reaction-arrow-elliptical-arc-arrow-filled-bow',
                    'reaction-arrow-elliptical-arc-arrow-filled-triangle',
                    'reaction-arrow-elliptical-arc-arrow-open-angle',
                    'reaction-arrow-elliptical-arc-arrow-open-half-angle',
                    'open',
                    'save',
                    'clear',
                    'shape',
                    'shape-ellipse',
                    'shape-rectangle',
                    'shape-line',
                    'fullscreen',
                    '3d-view',
                    'transform-rotate',
                    'transform-flip-h',
                    'transform-flip-v',
                ])}
            />
        );
    }
}

export default KetcherExample;
