import React, { useState, useRef, useEffect } from "react";
import {
  Upload,
  FileText,
  AlertTriangle,
  CheckCircle,
  Shield,
  Zap,
  Trash2,
  ChevronRight,
  Search,
  X,
  Database,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import Navbar from "./Navbar";
import Footer from "./Footer";
import Faq from "./Faq";
import { useNavigate } from "react-router-dom";
import axios from "axios";

export default function Home() {
  const [files, setFiles] = useState<File[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const navigate = useNavigate();

  const [activeFeature, setActiveFeature] = useState(0);
  const features = [
    "Real-time Detection",
    "Zero-day Protection",
    "Batch Processing",
    "Detailed Reports",
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveFeature((prev) => (prev + 1) % features.length);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleDrag = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) {
      setFiles(Array.from(files));
    }
  };

  const handleSubmit = async () => {
    if (files.length === 0) {
      alert("Please select a file first.");
      return;
    }

    setIsLoading(true);
    const formData = new FormData();
    formData.append("file", files[0]);

    try {
      console.log("Uploading file:", files[0]);
      const response = await fetch(
        "https://gajshield-flask-host.vercel.app/report",
        {
          method: "POST",
          body: formData,
          credentials: "include"
        }
      );

      const result = await response.json();

      if(result['engine_results'].length == 0){
        const ml_model_response = await fetch(
          "http://localhost:5000",
          {
            method: "POST",
            body: formData,
            credentials: "include"
          }
        );

        console.log("ML Model Response:", ml_model_response);
      }

      if (response.ok) {
        
        console.log("File uploaded successfully:", result);

        try {
          const response1 = await axios.post(
            "http://127.0.0.1:8080/report/chat",
            {
              prompt: result,
              method: "POST",
              body: formData,
              credentials: "include"
            }
          );
          if (response1.status >= 200 && response1.status < 300) {
            const result = response1.data;
            navigate("/result", { state: { result } });
            console.log("Result:", result);
          } else {
            console.error("Upload failed:", response1.statusText);
            alert("File upload failed!");
          }
        } catch (error) {
          console.error("Error uploading file:", error);
          alert("Error uploading file!");
        }
      } else {
        console.error("Upload failed:", response.statusText);
        alert("File upload failed!");
      }
    } catch (error) {
      console.error("Error uploading file:", error);
      alert("Error uploading file!");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <Navbar />
      <div className="flex flex-col min-h-screen bg-gradient-to-b from-background via-slate-50 to-background dark:from-background dark:via-slate-950 dark:to-background">
        <main className="flex-1">
          <section className="w-full py-20 relative overflow-hidden">
            <div className="container px-4 md:px-6 relative z-10">
              <div className="grid gap-8 lg:grid-cols-[1fr_450px] lg:gap-12 xl:grid-cols-[1fr_550px]">
                <div className="flex flex-col justify-center space-y-6">
                  <div className="inline-flex items-center px-3 py-1 text-sm font-medium text-primary bg-primary/10 rounded-full w-fit">
                    <Zap className="h-3.5 w-3.5 mr-1" />
                    <span>Next-Gen Security</span>
                  </div>
                  <div className="space-y-4">
                    <h1 className="text-4xl font-bold tracking-tighter sm:text-5xl xl:text-6xl/none bg-clip-text text-transparent bg-gradient-to-r from-blue-800 to-blue-600">
                      AI-Powered Malware Detection
                    </h1>
                    <p className="max-w-[600px] text-lg md:text-xl text-slate-700 dark:text-slate-300">
                      Protect your systems with our advanced AI that detects
                      known and unknown threats in real-time with unmatched
                      accuracy.
                    </p>
                  </div>
                  <div className="flex flex-col gap-3 min-[400px]:flex-row">
                    <Button
                      size="lg"
                      className="gap-2 bg-primary font-medium text-primary-foreground hover:bg-primary/90 shadow-lg shadow-primary/20 transition-all hover:shadow-xl hover:shadow-primary/30"
                      onClick={() => {
                        const scanSection =
                          document.getElementById("scan-section");
                        if (scanSection) {
                          scanSection.scrollIntoView({ behavior: "smooth" });
                        }
                      }}
                    >
                      <Shield className="h-5 w-5" />
                      Start Scanning
                    </Button>
                    <Button
                      size="lg"
                      variant="outline"
                      className="border-slate-300 text-slate-900 dark:border-slate-700 dark:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-800"
                    >
                      <Search className="h-5 w-5 mr-2" />
                      Learn More
                    </Button>
                  </div>
                  <div className="flex items-center gap-4 text-sm text-muted-foreground mt-6">
                    <div className="flex items-center">
                      <CheckCircle className="h-4 w-4 text-green-500 mr-1.5" />
                      High Detection Rate
                    </div>
                    <div className="flex items-center">
                      <CheckCircle className="h-4 w-4 text-green-500 mr-1.5" />
                      Real-time Protection
                    </div>
                    <div className="flex items-center">
                      <CheckCircle className="h-4 w-4 text-green-500 mr-1.5" />
                      Enterprise Ready
                    </div>
                  </div>
                </div>
                <div
                  className="mx-auto flex w-full items-center justify-center"
                  id="scan-section"
                >
                  <Card className="w-full border border-slate-200 dark:border-slate-800 shadow-xl hover:shadow-2xl transition-all bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm">
                    <CardHeader className="pb-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <CardTitle className="text-2xl flex items-center gap-2">
                            <Shield className="h-5 w-5 text-primary" />
                            Quick Scan
                          </CardTitle>
                          <CardDescription className="text-sm mt-1">
                            Upload a file to check for malware
                          </CardDescription>
                        </div>
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary/10 text-primary">
                          Secure
                        </span>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div
                        className={`flex flex-col items-center justify-center border-2 border-dashed rounded-xl p-10 space-y-3 transition-all ${
                          dragActive
                            ? "border-primary bg-primary/5"
                            : "border-slate-200 dark:border-slate-800"
                        }`}
                        onDragEnter={handleDrag}
                        onDragLeave={handleDrag}
                        onDragOver={handleDrag}
                        onDrop={handleDrop}
                      >
                        {files.length === 0 ? (
                          <>
                            <div className="h-20 w-20 rounded-full bg-primary/10 flex items-center justify-center">
                              <Upload className="h-10 w-10 text-primary" />
                            </div>
                            <p className="text-center">
                              <span className="font-medium">
                                Drop files here
                              </span>{" "}
                              or click to browse
                            </p>
                            <p className="text-xs text-muted-foreground max-w-xs text-center">
                              Supports .exe, .csv, .docx, .bat and other file
                              types
                            </p>
                            <input
                              type="file"
                              id="file-upload"
                              className="hidden"
                              onChange={handleFileChange}
                              ref={fileInputRef}
                            />
                            <Button
                              variant="outline"
                              className="mt-2"
                              onClick={() => fileInputRef.current?.click()}
                            >
                              Select File
                            </Button>
                          </>
                        ) : (
                          <div className="w-full">
                            <div className="flex items-center gap-3 p-3 bg-slate-50 dark:bg-slate-900 rounded-lg">
                              <div className="h-10 w-10 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center">
                                <FileText className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                              </div>
                              <div className="flex-1 min-w-0">
                                <p className="font-medium truncate">
                                  {files[0].name}
                                </p>
                                <p className="text-xs text-muted-foreground">
                                  {(files[0].size / 1024).toFixed(1)} KB
                                </p>
                              </div>
                              <Button
                                variant="ghost"
                                size="icon"
                                onClick={() => setFiles([])}
                                className="h-8 w-8"
                              >
                                <X className="h-4 w-4" />
                              </Button>
                            </div>
                          </div>
                        )}
                      </div>

                      {files.length > 0 && (
                        <div className="flex gap-3 mt-4">
                          <Button
                            variant="outline"
                            className="flex-1"
                            onClick={() => setFiles([])}
                            disabled={isLoading}
                          >
                            <Trash2 className="h-4 w-4 mr-2" />
                            Remove
                          </Button>
                          <Button
                            className="flex-1 bg-black text-white gap-2 hover:bg-black/90 shadow-lg shadow-black/20 transition-all hover:shadow-xl hover:shadow-black/30"
                            onClick={handleSubmit}
                            disabled={isLoading}
                          >
                            {isLoading ? (
                              <>
                                <div className="animate-spin h-4 w-4 border-2 border-current border-t-transparent rounded-full" />
                                Scanning...
                              </>
                            ) : (
                              <>
                                <Zap className="h-4 w-4 hover:bg-black" />
                                Scan Now
                              </>
                            )}
                          </Button>
                        </div>
                      )}
                    </CardContent>
                    <CardFooter className="border-t pt-4 flex justify-between">
                      <p className="text-xs text-muted-foreground">
                        Your files are analyzed securely
                      </p>
                      <p className="text-xs font-medium text-primary">
                        256-bit Encrypted
                      </p>
                    </CardFooter>
                  </Card>
                </div>
              </div>
            </div>
          </section>

          {/* Features Section */}
          <section className="w-full py-16 md:py-24 lg:py-32 bg-slate-50 dark:bg-slate-900/50">
            <div className="container px-4 md:px-6">
              <div className="mx-auto flex max-w-[58rem] flex-col items-center justify-center gap-4 text-center">
                <span className="inline-flex items-center px-3 py-1 text-sm font-medium bg-primary/10 text-primary rounded-full">
                  <Shield className="h-3.5 w-3.5 mr-1" />
                  Advanced Protection
                </span>
                <h2 className="text-3xl font-bold leading-[1.1] sm:text-3xl md:text-5xl">
                  Comprehensive Security{" "}
                  <span className="text-primary">Features</span>
                </h2>
                <p className="max-w-[85%] leading-normal text-muted-foreground sm:text-lg sm:leading-7">
                  Our AI-powered malware detection system provides multi-layered
                  protection against sophisticated threats.
                </p>
              </div>

              <div className="mx-auto grid justify-center gap-6 sm:grid-cols-2 md:grid-cols-4 lg:gap-10 mt-16">
                {features.map((feature, index) => (
                  <Card
                    key={index}
                    className={`overflow-hidden group transition-all duration-300 hover:shadow-xl ${
                      activeFeature === index
                        ? "border-primary ring-2 ring-primary/20"
                        : "border-slate-200 dark:border-slate-800"
                    }`}
                  >
                    <div
                      className={`h-2 bg-gradient-to-r ${
                        index === 0
                          ? "from-primary to-blue-500"
                          : index === 1
                          ? "from-purple-500 to-pink-500"
                          : index === 2
                          ? "from-amber-500 to-orange-500"
                          : "from-green-500 to-emerald-500"
                      }`}
                    ></div>
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-lg">{feature}</CardTitle>
                        <div
                          className={`h-8 w-8 rounded-full flex items-center justify-center ${
                            index === 0
                              ? "bg-primary/10"
                              : index === 1
                              ? "bg-purple-500/10"
                              : index === 2
                              ? "bg-amber-500/10"
                              : "bg-green-500/10"
                          }`}
                        >
                          {index === 0 ? (
                            <Zap className="h-4 w-4 text-primary" />
                          ) : index === 1 ? (
                            <Shield className="h-4 w-4 text-purple-500" />
                          ) : index === 2 ? (
                            <Database className="h-4 w-4 text-amber-500" />
                          ) : (
                            <FileText className="h-4 w-4 text-green-500" />
                          )}
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-muted-foreground">
                        {index === 0
                          ? "Instantly analyze files for malicious content using advanced AI algorithms."
                          : index === 1
                          ? "Identify unknown threats before they can cause damage to your systems."
                          : index === 2
                          ? "Scan multiple files simultaneously for efficient threat detection."
                          : "Get comprehensive analysis reports with actionable insights."}
                      </p>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          </section>

          {/* How It Works Section */}
          <section className="w-full py-16 md:py-24 lg:py-32 relative overflow-hidden">
            {/* Background blob */}
            <div className="absolute inset-0 overflow-hidden opacity-10">
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-primary/30 rounded-full blur-3xl" />
            </div>

            <div className="container px-4 md:px-6 relative z-10">
              <div className="mx-auto flex max-w-[58rem] flex-col items-center justify-center gap-4 text-center">
                <span className="inline-flex items-center px-3 py-1 text-sm font-medium bg-primary/10 text-primary rounded-full">
                  <CheckCircle className="h-3.5 w-3.5 mr-1" />
                  Simple Process
                </span>
                <h2 className="text-3xl font-bold leading-[1.1] sm:text-3xl md:text-5xl">
                  How It <span className="text-primary">Works</span>
                </h2>
                <p className="max-w-[85%] leading-normal text-muted-foreground sm:text-lg sm:leading-7">
                  Our AI-powered system uses advanced machine learning to detect
                  malware with high accuracy.
                </p>
              </div>

              <div className="mx-auto grid max-w-5xl items-center gap-10 py-12 lg:grid-cols-2 lg:gap-12">
                <div className="flex flex-col justify-center space-y-6">
                  <ul className="grid gap-8">
                    {[
                      {
                        icon: <Upload className="h-5 w-5" />,
                        title: "Upload Files",
                        description:
                          "Upload individual files or batch process multiple files at once.",
                        color: "bg-primary/10 text-primary",
                      },
                      {
                        icon: <FileText className="h-5 w-5" />,
                        title: "AI Analysis",
                        description:
                          "Our AI engine analyzes file structure, behavior patterns, and code signatures.",
                        color: "bg-purple-500/10 text-purple-500",
                      },
                      {
                        icon: <CheckCircle className="h-5 w-5" />,
                        title: "Threat Detection",
                        description:
                          "Receive instant results with detailed information about detected threats.",
                        color: "bg-green-500/10 text-green-500",
                      },
                    ].map((step, index) => (
                      <li key={index} className="flex items-start gap-4 group">
                        <div
                          className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-xl ${step.color} shadow-sm group-hover:shadow-md transition-all`}
                        >
                          {step.icon}
                        </div>
                        <div className="grid gap-1">
                          <h3 className="text-xl font-bold">{step.title}</h3>
                          <p className="text-muted-foreground">
                            {step.description}
                          </p>
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="mx-auto flex w-full items-center justify-center">
                  <Tabs defaultValue="upload" className="w-full max-w-md">
                    <TabsList className="grid w-full grid-cols-3 bg-slate-100 dark:bg-slate-800 p-1 rounded-xl">
                      {["upload", "analyze", "results"].map((tab, index) => (
                        <TabsTrigger
                          key={tab}
                          value={tab}
                          className="rounded-lg data-[state=active]:bg-white dark:data-[state=active]:bg-slate-950 data-[state=active]:shadow-sm transition-all capitalize"
                        >
                          {tab}
                        </TabsTrigger>
                      ))}
                    </TabsList>

                    <div className="mt-6 border border-slate-200 dark:border-slate-800 rounded-xl overflow-hidden shadow-xl">
                      <TabsContent value="upload" className="m-0">
                        <div className="flex flex-col items-center justify-center space-y-4 p-8 bg-slate-50 dark:bg-slate-900">
                          <div className="h-20 w-20 rounded-full bg-primary/10 flex items-center justify-center">
                            <Upload className="h-10 w-10 text-primary" />
                          </div>
                          <h3 className="text-lg font-medium">Upload File</h3>
                          <p className="text-center text-sm text-muted-foreground max-w-xs">
                            Drag and drop files here or click to browse your
                            device
                          </p>
                          <Button variant="outline" size="sm">
                            Select File
                          </Button>
                        </div>
                      </TabsContent>

                      <TabsContent value="analyze" className="m-0">
                        <div className="flex flex-col items-center justify-center space-y-4 p-8 bg-slate-50 dark:bg-slate-900">
                          <div className="relative h-20 w-20 flex items-center justify-center">
                            <div className="absolute inset-0 rounded-full border-4 border-t-purple-500 border-purple-500/20 animate-spin" />
                            <FileText className="h-8 w-8 text-purple-500" />
                          </div>
                          <h3 className="text-lg font-medium">AI Analysis</h3>
                          <p className="text-center text-sm text-muted-foreground max-w-xs">
                            Our AI is analyzing file structure and behavior
                            patterns...
                          </p>
                          <div className="w-full max-w-xs bg-slate-200 dark:bg-slate-800 h-2 rounded-full overflow-hidden">
                            <div className="h-full bg-purple-500 w-2/3 rounded-full animate-pulse" />
                          </div>
                        </div>
                      </TabsContent>

                      <TabsContent value="results" className="m-0">
                        <div className="flex flex-col items-center justify-center space-y-4 p-8 bg-slate-50 dark:bg-slate-900">
                          <div className="h-20 w-20 rounded-full bg-red-500/10 flex items-center justify-center">
                            <AlertTriangle className="h-10 w-10 text-red-500" />
                          </div>
                          <h3 className="text-lg font-bold text-red-500">
                            Malware Detected
                          </h3>
                          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-3">
                            <p className="text-center text-sm text-red-700 dark:text-red-300">
                              Trojan detected in file. We recommend immediate
                              deletion.
                            </p>
                          </div>
                          <Button
                            className="mt-2 bg-red-500 hover:bg-red-600 text-white"
                            size="sm"
                          >
                            View Detailed Report
                          </Button>
                        </div>
                      </TabsContent>
                    </div>
                  </Tabs>
                </div>
              </div>
            </div>
          </section>
        </main>
        <Faq />
        <Footer />
      </div>
    </>
  );
}
