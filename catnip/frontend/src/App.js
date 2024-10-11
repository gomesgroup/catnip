import React from 'react';
import {BrowserRouter as Router, Route, Routes} from 'react-router-dom';
import SubmitPage from './pages/SubmitPage';
import RunPage from './pages/RunPage';
import {ThemeProvider} from "@mui/material/styles";
import theme from "./theme";

function App() {
    return (
        <ThemeProvider theme={theme}>
            <Router>
                <Routes>
                    <Route path="/" element={<SubmitPage/>}/>
                    <Route path="/run/:id" element={<RunPage/>}/>
                </Routes>
            </Router>
        </ThemeProvider>
    );
}

export default App;
