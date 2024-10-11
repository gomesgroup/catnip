import React, {useState} from 'react';
import {Paper, TableContainer, Table, TableHead, TableRow, TableBody, TableCell, Button} from "@mui/material";

function CollapsibleTable({step}) {
    const [isExpanded, setIsExpanded] = useState(false);

    const toggleExpand = () => {
        setIsExpanded(!isExpanded);
    };

    return (
        <>
            <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
                <TableContainer component={Paper} style={{maxHeight: isExpanded ? 'none' : '300px', overflow: 'auto'}}>
                    <Table size="small" stickyHeader>
                        <TableHead>
                            <TableRow>
                                <TableCell><b>Feature</b></TableCell>
                                <TableCell><b>Value</b></TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {Object.entries(step.table).map(([key, value]) => (
                                <TableRow key={key}>
                                    <TableCell>{value[0]}</TableCell>
                                    <TableCell>{value[1]}</TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </TableContainer>
                <Button variant="info" onClick={toggleExpand} sx={{marginTop: 3}}>
                    {isExpanded ? 'Collapse' : 'Expand'}
                </Button>
            </div>
        </>
    );
}

export default CollapsibleTable;
