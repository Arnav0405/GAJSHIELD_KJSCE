const express = require("express");
const app = express();
const mongoose = require("mongoose");
const { User } = require("./models/model.js");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken"); 
const cors = require("cors");
const port = 8080;
const reportRouter = require("./GroqReport.js");
const assistanceRouter = require("./GroqChat.js");


app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(cors({
    origin: ["http://localhost:5173", "http://localhost:5000","https://gajshield-flask-host.vercel.app"],
    credentials: true,
}));
async function main() {
    await mongoose.connect('mongodb+srv://manik:Tw0yDTt60qwOmGzO@cluster0.px6sa.mongodb.net/databaseName?retryWrites=true&w=majority') // Add database name
        .then(() => {
            console.log("Connected to Database");
        });
}
main();

app.listen(port, () => {
    console.log(`server is listening on ${port}.`);
});

app.post("/login", async (req, res) => {
    const { username, password } = req.body;

    try {
        const user = await User.findOne({ username });
        if (!user) {
            return res.status(404).json({ message: "User not found" });
        }

        const validPassword = await bcrypt.compare(password, user.password);
        if (!validPassword) {
            return res.status(401).json({ message: 'Invalid credentials' });
        }
        const token = jwt.sign({ userId: user._id }, 'your_secret_key', { expiresIn: '1h' });
        res.cookie('token', token, { httpOnly: true , sameSite: 'lax', secure: true });
        res.cookie('username', username, { httpOnly: false , sameSite: 'lax', secure: true});
        res.json({ message: 'Logged in successfully' });
    } catch (error) {
        console.error(error);
        res.status(500).send("Internal Server Error");
    }
});

app.post("/signup", async (req, res) => {
    const { username, email, password } = req.body;

    try {
        const existingUser = await User.findOne({ username });
        if (existingUser) {
            return res.status(400).json({ message: "User already exists" });
        }

        // Hash password
        const salt = await bcrypt.genSalt(10);
        const hashedPassword = await bcrypt.hash(password, salt);

        // Create new user
        const newUser = new User({
            email,
            username,
            password: hashedPassword // Store hashed password
        });

        await newUser.save();
        res.status(201).json({ message: "User created successfully" });
    } catch (error) {
        console.error(error);
        res.status(500).json({ message: "Internal Server Error" });
    }
});

app.post("/logout", (req, res) => { 
    res.clearCookie('token');
    res.clearCookie('username');
    res.json({ message: 'Logged out successfully' });
});

app.use("/report", reportRouter);
app.use("/assistance", assistanceRouter);