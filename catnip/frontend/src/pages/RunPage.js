import React, {useState, useEffect} from "react";
import {useParams} from "react-router-dom";
import AppBarComponent from "../components/AppBarComponent";
import MoleculeStructure from "../components/MoleculeStructure/MoleculeStructure";
import CopyableText from "../components/CopyableText";
import MolecularViewer from "../components/MolecularViewer3D";
import TableCollapse from "../components/TableCollapse";
import DownloadButton from "../components/DownloadButton";
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
import {data} from "../data";
import ScatterPlot from "../components/ScatterPlot";

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

const RunPage = () => {
    const {id: runId} = useParams();
    const [state, setState] = useState({
        smilesData: null,
        statusData: [],
        overallStatus: "",
        tabValue: 0
    });

    const POLLING_INTERVAL = 500; // 5 seconds

    useEffect(() => {
        const fetchMoleculeData = async () => {
            try {
                const response = await fetch(``); // TODO: Add the URL to fetch the molecule data
                const result = await response.json();

                setState(prevState => ({
                    ...prevState,
                    smilesData: result.data,
                    statusData: result.status,
                    pca_weights: result.pca_weights,
                    neighbors: result.neighbors,
                    overallStatus: result.overall_status,
                    ranking: result.ranking,
                    columns: Object.keys(result.ranking?.[0] ?? {}),
                    exact_match: result.exact_match?.found ?? false,
                    exact_match_table: result.exact_match?.seqs ?? [],
                    exact_match_table_columns: Object.keys(result.exact_match?.seqs?.[0] ?? {}),
                }));

            } catch (error) {
                console.error("Failed to fetch molecule data:", error);
            }
        };

        fetchMoleculeData();

        const interval = setInterval(() => {
            if (state.overallStatus === "done") {
                clearInterval(interval);
            } else {
                fetchMoleculeData();
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
            {step.data_type === "molecule" ? (
                step.molecule ? (
                    <div id="128" className="mol-container">
                        <MolecularViewer format="sdf" moleculeData={step.molecule}/>
                    </div>
                ) : (
                    <Skeleton variant="rectangular" width="100%" animation="wave" height={400}/>
                )
            ) : (
                step.table ? (
                    <TableCollapse step={step}/>
                ) : (
                    step.status === "complete" ? (<DownloadButton runId={runId}/>) : (<></>)
                )
            )}
        </Grid>
    );

    const handleTabChange = (event, newValue) => {
        setState((prevState) => ({...prevState, tabValue: newValue}));
    };

    console.log(state.tabValue)
    return (
        <div>
            <AppBarComponent/>
            <Container maxWidth="xl" sx={{marginTop: "128px"}}>
                <Grid container spacing={8}>
                    <Grid item xs={5}>
                        {state.smilesData ? (
                            <MoleculeStructure
                                id={runId}
                                structure={state.smilesData}
                                height={200}
                                width={200}
                                svgMode
                            />
                        ) : (
                            <p>Loading...</p>
                        )}
                        <Typography variant="h5" sx={{marginTop: 3, marginBottom: 1}}>
                            <CopyableText text={runId}/>
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
            <Box sx={{backgroundColor: "black", color: "white", paddingTop: "64px"}}>
                <Container maxWidth="xl">
                    <Tabs value={state.tabValue} onChange={handleTabChange} aria-label="basic tabs example"
                          sx={{backgroundColor: "white"}}>
                        <Tab label="PCA & Components" {...a11yProps(0)} />
                        <Tab label="Ranking" {...a11yProps(1)} />
                        <Tab label="Exact Match" {...a11yProps(2)} />
                    </Tabs>
                    <TabPanel value={state.tabValue} index={0}>
                        <Grid container spacing={8}>
                            <Grid item xs={6}>
                                <Typography variant="h6" sx={{color: "white", marginBottom: 3}}>
                                    Substrate space
                                </Typography>
                                <Box sx={{
                                    width: '100%',
                                    height: 0,
                                    paddingBottom: '100%',
                                    position: 'relative'
                                }}>
                                    <div style={{position: 'absolute', top: 0, left: 0, right: 0, bottom: 0}}>
                                        <ScatterPlot data={data} largeDotCoordinates={state.pca_weights}
                                                     largeDotLegend={state.smilesData}/>
                                    </div>
                                </Box>
                                <br/>
                            </Grid>
                            <Grid item xs={6}>
                                <Typography variant="h6" sx={{color: "white", marginBottom: "64px"}}>
                                    Results
                                </Typography>
                                {state.pca_weights && (
                                    <TableContainer component={Paper} sx={{maxHeight: 440}}>
                                        <Table stickyHeader aria-label="pca weights table">
                                            <TableHead>
                                                <TableRow>
                                                    <TableCell>Component</TableCell>
                                                    <TableCell align="right">Weight</TableCell>
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>
                                                {state.pca_weights.map((weight, index) => (
                                                    <TableRow key={index}>
                                                        <TableCell component="th" scope="row">
                                                            Component {index + 1}
                                                        </TableCell>
                                                        <TableCell align="right">{weight.toFixed(2)}</TableCell>
                                                    </TableRow>
                                                ))}
                                            </TableBody>
                                        </Table>
                                    </TableContainer>
                                )}
                                <br/>
                                {state.neighbors && (<TableContainer component={Paper} sx={{maxHeight: 440}}>
                                        <Table stickyHeader aria-label="neighbors table">
                                            <TableHead>
                                                <TableRow>
                                                    {state.neighbors.map((_, index) => (
                                                        <TableCell key={index}>N{index + 1}</TableCell>
                                                    ))}
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>
                                                <TableRow>
                                                    {state.neighbors.map((neighbor, index) => (
                                                        <TableCell key={index}>{neighbor}</TableCell>
                                                    ))}
                                                </TableRow>
                                            </TableBody>
                                        </Table>
                                    </TableContainer>
                                )}
                            </Grid>
                        </Grid>
                    </TabPanel>
                    <TabPanel value={state.tabValue} index={1}>
                        <Grid container spacing={8}>
                            <Grid item xs={12}>
                                <Typography variant="h6" sx={{color: "white", marginBottom: 3}}>
                                    Ranking
                                </Typography>
                                {state.ranking && (
                                    <TableContainer component={Paper}>
                                        <Table sx={{minWidth: 650}} aria-label="simple table">
                                            <TableHead>
                                                <TableRow>
                                                    {state.columns.map((column, index) => (
                                                        <TableCell key={index}>{column}</TableCell>
                                                    ))}
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>
                                                {state.ranking.map((row, rowIndex) => (
                                                    <TableRow key={rowIndex}>
                                                        {state.columns.map((column, colIndex) => (
                                                            <TableCell key={colIndex}>{row[column]}</TableCell>
                                                        ))}
                                                    </TableRow>
                                                ))}
                                            </TableBody>
                                        </Table>
                                    </TableContainer>
                                )}
                            </Grid>
                        </Grid>
                    </TabPanel>
                    <TabPanel value={state.tabValue} index={2}>
                        <Grid container spacing={8}>
                            <Grid item xs={12}>
                                <Typography variant={"h6"} sx={{color: "white", marginBottom: 3}}>
                                    Exact Match
                                </Typography>
                                {state.exact_match && (
                                    <TableContainer component={Paper}>
                                        <Table sx={{minWidth: 650}} aria-label="simple table">
                                            <TableHead>
                                                <TableRow>
                                                    {state.exact_match_table_columns.map((column, index) => (
                                                        <TableCell key={index}>{column}</TableCell>
                                                    ))}
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>
                                                {state.exact_match_table.map((row, rowIndex) => (
                                                    <TableRow key={rowIndex}>
                                                        {state.exact_match_table_columns.map((column, colIndex) => (
                                                            <TableCell key={colIndex}>{row[column]}</TableCell>
                                                        ))}
                                                    </TableRow>
                                                ))}
                                            </TableBody>
                                        </Table>
                                    </TableContainer>
                                )}
                                <br></br>
                            </Grid>
                        </Grid>
                    </TabPanel>
                </Container>
            </Box>
        </div>
    );
};

export default RunPage;
