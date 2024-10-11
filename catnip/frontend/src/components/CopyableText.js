import React, {useState} from 'react';
import FileCopyIcon from '@mui/icons-material/FileCopy';
import Snackbar from '@mui/material/Snackbar';
import IconButton from '@mui/material/IconButton';

function CopyableText({text}) {
    const [copied, setCopied] = useState(false);

    const handleCopyClick = () => {
        navigator.clipboard.writeText(text).then(() => {
            setCopied(true);
            setTimeout(() => {
                setCopied(false);
            }, 2500);
        });
    };

    return (
        <div style={{display: 'flex', alignItems: 'center'}}>
            <span>{text}</span>
            <IconButton onClick={handleCopyClick} size="small">
                <FileCopyIcon/>
            </IconButton>
            <Snackbar
                open={copied}
                autoHideDuration={2000}
                message="Copied!"
                anchorOrigin={{vertical: 'bottom', horizontal: 'right'}}
            />
        </div>
    );
}

export default CopyableText;
