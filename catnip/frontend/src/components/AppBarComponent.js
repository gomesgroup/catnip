import React from 'react';
import {AppBar, Toolbar, Typography, Button} from '@mui/material';
import {Link} from 'react-router-dom';

function AppBarComponent() {
    return (
        <AppBar position="static" sx={{height: "48px", backgroundColor: "black"}}>
            <Toolbar>
                <Button component={Link} to="/" variant="text" color="inherit">
                    🌿catnip
                </Button>
                <Typography variant="caption" component="div" sx={{flexGrow: 1, marginLeft: 1}}>
                    Carnegie Mellon University & University of Michigan
                </Typography>
            </Toolbar>
        </AppBar>
    );
}

export default AppBarComponent;
