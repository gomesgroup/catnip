import React, {useState, useEffect} from "react";
import {useParams} from "react-router-dom";
import AppBarComponent from "../components/AppBarComponent";
import CopyableText from "../components/CopyableText";
import EnzymeDownloadButton from "../components/EnzymeDownloadButton";
import MoleculeStructure from "../components/MoleculeStructure/MoleculeStructure";
import {
    Paper,
    TableContainer,
    Table,
    TableHead,
    TableRow,
    TableBody,
    TableCell,
    Typography,
    Skeleton,
    Box,
    LinearProgress,
    Grid,
    Container,
    Tab,
    Tabs,
} from "@mui/material";

const TabPanel = (props) => {
    const {children, value, index, ...other} = props;

    return (
        <div
            role="tabpanel"
            hidden={value !== index}
            id={`simple-tabpanel-${index}`}
            aria-labelledby={`simple-tab-${index}`}
            {...other}
        >
            {value === index && <Box sx={{p: 3}}>{children}</Box>}
        </div>
    );
};

const a11yProps = (index) => {
    return {
        id: `simple-tab-${index}`,
        'aria-controls': `simple-tabpanel-${index}`,
    };
};

const EnzymeRunPage = () => {
    const {id: runId} = useParams();
    const [state, setState] = useState({
        sequence: null,
        statusData: [],
        overallStatus: "",
        tabValue: 0,
        substrates: []
    });

    const POLLING_INTERVAL = 500; // 0.5 seconds

    useEffect(() => {
        const fetchEnzymeData = async () => {
            try {
                const response = await fetch(`/api/enzyme-run/${runId}`);
                const result = await response.json();

                setState(prevState => ({
                    ...prevState,
                    sequence: result.sequence,
                    statusData: result.status,
                    overallStatus: result.overall_status,
                    substrates: result.substrates || []
                }));

            } catch (error) {
                console.error("Failed to fetch enzyme data:", error);
            }
        };

        fetchEnzymeData();

        const interval = setInterval(() => {
            if (state.overallStatus === "done") {
                clearInterval(interval);
            } else {
                fetchEnzymeData();
            }
        }, POLLING_INTERVAL);

        return () => clearInterval(interval);
    }, [runId, state.overallStatus]);

    const renderProgress = (step) => {
        const progressProps = {
            variant: "determinate",
            color: "success",
            value: 100,
        };

        switch (step.status) {
            case "in_progress":
                if (step.progress >= 0) {
                    progressProps.value = step.progress;
                } else {
                    delete progressProps.value;
                    delete progressProps.variant;
                }
                break;
            case "pending":
                progressProps.color = "inherit";
                delete progressProps.value;
                delete progressProps.variant;
                break;
            case "error":
                progressProps.color = "error";
                break;
            default:
                break;
        }

        return <LinearProgress {...progressProps} />;
    };

    const renderStep = (step, index) => (
        <Grid item xs={3} key={index}>
            <Typography variant="h6" sx={{marginTop: 3}}>
                <b>{step.name}</b>
            </Typography>
            <Typography variant="body1" sx={{marginBottom: 1}}>
                {step.description}
            </Typography>
            {renderProgress(step)}
            <br/>
            {step.table && step.status === "complete" ? (
                <TableContainer component={Paper} sx={{maxHeight: 300}}>
                    <Table stickyHeader size="small">
                        <TableHead>
                            <TableRow>
                                {Object.keys(step.table[0] || {}).map((key) => (
                                    <TableCell key={key}>{key}</TableCell>
                                ))}
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {step.table.map((row, i) => (
                                <TableRow key={i}>
                                    {Object.entries(row).map(([key, value]) => (
                                        <TableCell key={key}>{value}</TableCell>
                                    ))}
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </TableContainer>
            ) : (
                step.status === "complete" && step.name === "Step 4" ?
                    (<EnzymeDownloadButton runId={runId}/>) :
                    (<Skeleton variant="rectangular" width="100%" animation="wave" height={200}/>)
            )}
        </Grid>
    );

    const handleTabChange = (event, newValue) => {
        setState((prevState) => ({...prevState, tabValue: newValue}));
    };

    return (
        <div>
            <AppBarComponent/>
            <Container maxWidth="xl" sx={{marginTop: "128px"}}>
                <Grid container spacing={8}>
                    <Grid item xs={5}>
                        <Typography variant="h5">
                            Enzyme Sequence Analysis
                        </Typography>
                        {state.sequence ? (
                            <Box
                                sx={{
                                    marginTop: 2,
                                    maxHeight: '150px',
                                    overflow: 'auto',
                                    fontFamily: 'monospace',
                                    bgcolor: '#f5f5f5',
                                    p: 2,
                                    borderRadius: 1
                                }}
                            >
                                {state.sequence}
                            </Box>
                        ) : (
                            <Skeleton variant="rectangular" width="100%" height={150}/>
                        )}
                        <Typography variant="h6" sx={{marginTop: 3, marginBottom: 1}}>
                            Run ID: <CopyableText text={runId}/>
                        </Typography>
                        <Typography variant="body1" sx={{marginBottom: 3}}>
                            Note that the results will be stored only for 7 days. Please
                            save the run ID to access the results later.
                        </Typography>
                    </Grid>
                    <Grid item xs={7}></Grid>
                </Grid>
            </Container>
            <hr/>
            <Container maxWidth="2560px" sx={{marginBottom: 3}}>
                <Grid container spacing={8}>
                    {state.statusData.map(renderStep)}
                </Grid>
            </Container>
            <Box sx={{backgroundColor: "black", color: "white", paddingTop: "64px", paddingBottom: "64px"}}>
                <Container maxWidth="xl">
                    <Tabs value={state.tabValue} onChange={handleTabChange} aria-label="results tabs"
                          sx={{backgroundColor: "white"}}>
                        <Tab label="Substrate Predictions" {...a11yProps(0)} />
                    </Tabs>
                    <TabPanel value={state.tabValue} index={0}>
                        <Typography variant="h6" sx={{color: "white", marginBottom: "24px"}}>
                            Predicted Compatible Substrates
                        </Typography>
                        {state.substrates && state.substrates.length > 0 ? (
                            <TableContainer component={Paper} sx={{maxHeight: 600}}>
                                <Table stickyHeader aria-label="substrates table">
                                    <TableHead>
                                        <TableRow>
                                            <TableCell>Structure</TableCell>
                                            <TableCell>SMILES</TableCell>
                                            <TableCell>Enzyme Neighbor</TableCell>
                                            <TableCell>Substrate</TableCell>
                                            <TableCell>Score</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {state.substrates.map((substrate, index) => (
                                            <TableRow key={index}>
                                                <TableCell>
                                                    <MoleculeStructure
                                                        id={`smiles-${index}`}
                                                        structure={substrate.smiles}
                                                        width={150}
                                                        height={120}
                                                        svgMode={true}
                                                    />
                                                </TableCell>
                                                <TableCell>
                                                    <CopyableText text={substrate.smiles}/>
                                                </TableCell>
                                                <TableCell>{substrate.enzyme_neighbor}</TableCell>
                                                <TableCell>{substrate.substrate}</TableCell>
                                                <TableCell>{substrate.score}</TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </TableContainer>
                        ) : (
                            state.overallStatus === "done" ? (
                                <Typography variant="body1" sx={{color: "white"}}>
                                    No compatible substrates found.
                                </Typography>
                            ) : (
                                <Skeleton variant="rectangular" width="100%" height={400}
                                          sx={{bgcolor: 'rgba(255, 255, 255, 0.1)'}}/>
                            )
                        )}
                    </TabPanel>
                </Container>
            </Box>
        </div>
    );
};

export default EnzymeRunPage; 
