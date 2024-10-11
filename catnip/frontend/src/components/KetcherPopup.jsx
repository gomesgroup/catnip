import React, {useState} from 'react';

import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import Slide from '@mui/material/Slide';

import {ChemicalMimeType} from 'ketcher-core';
import {StandaloneStructServiceProvider} from 'ketcher-standalone';
import {Editor} from 'ketcher-react';
import "ketcher-react/dist/index.css";

const structServiceProvider = new StandaloneStructServiceProvider();

const Transition = React.forwardRef(function Transition(props, ref) {
    return <Slide direction="up" ref={ref} {...props} />;
});

export default function FullScreenDialog({ketcherCallBack}) {
    const [open, setOpen] = React.useState(false);
    const [ketcher, setKetcher] = useState();

    const handleClickOpen = () => {
        setOpen(true);
    };

    async function handleClose() {
        let smiles = await ketcher.getSmiles().then(result => {
            return result;
        });
        let mol = await ketcher.indigo.aromatize(smiles).then(result => {
            return result;
        });
        smiles = await ketcher.indigo.convert(mol, {outputFormat: ChemicalMimeType.DaylightSmiles}).then(result => {
            return result.struct
        })
        const SMARTS = await ketcher.getSmarts().then(result => {
            return result;
        });
        ketcherCallBack([smiles, SMARTS]);
        setOpen(false);
    };

    return (
        <div>
            <Button variant="contained" sx={{mb: 5}} onClick={handleClickOpen}>
                Open Molecular Sketcher
            </Button>
            <Dialog
                fullWidth={true}
                maxWidth={"lg"}
                open={open}
                onClose={handleClose}
                keepMounted
                TransitionComponent={Transition}
            >
                <Editor
                    staticResourcesUrl={process.env.PUBLIC_URL}
                    structServiceProvider={structServiceProvider}
                    onInit={(ketcher) => {
                        setKetcher(ketcher)
                    }}
                />
            </Dialog>
        </div>
    );
}
