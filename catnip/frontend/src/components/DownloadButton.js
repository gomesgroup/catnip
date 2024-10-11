import React from 'react';
import Button from '@mui/material/Button';

function DownloadButton({runId}) {
    const handleDownload = () => {
        if (!runId) {
            console.error('Run ID is undefined');
            return;
        }

        const url = ``; // TODO: Add the URL to download the CSV file

        fetch(url)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.blob();
            })
            .then(blob => {
                const blobUrl = window.URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = blobUrl;
                link.download = `run_${runId}_features.csv`;
                link.click();
                window.URL.revokeObjectURL(blobUrl);
            })
            .catch(e => console.error('Error during file download:', e));
    };

    return (
        <Button variant="contained" color="primary" onClick={handleDownload}>
            Download CSV
        </Button>
    );
}

export default DownloadButton;
