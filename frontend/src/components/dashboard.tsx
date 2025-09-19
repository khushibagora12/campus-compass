import ChatInterface from "./chatbot/interface";
import ShinyText from "./ShinyText"
import StarBorder from './StarBorder'
import { useState } from "react";

export default function Dashboard() {
    const [toggle, setToggle] = useState(false)
    return (
        <>
            <div className="h-screen overflow-hidden">
                <nav>
                    <div className='text-[20px] xl:text-3xl text-shadow-lg text-black fixed top-8 xl:top-5 left-5 font-cinzel'>Campus Compass</div>
                    <ChatInterface toggle={toggle} setToggle={setToggle}/>
                </nav>
                <div className='w-full h-[500px] flex justify-center items-center text-center'>
                    <div>
                        <div className='justify-center items-center text-5xl lg:text-6xl font-italiana'>Ask Away Your <br />
                            <span className="">
                                <ShinyText
                                    text="Queries"
                                    disabled={false}
                                    speed={10}
                                    className='custom-class'
                                />
                            </span>
                        </div>
                        <div className='tracking-wider lg:text-lg m-5 text-gray-700 '>Ask anything about your institute - <br /> quick, simple, and reliable.</div>
                        <div className="">
                            <StarBorder
                                as="button"
                                onClick={() => setToggle(!toggle)}
                                className="custom-class"
                                color="#7C00FE"
                                thickness={6}
                                speed="5s"
                            >
                                Ask query
                            </StarBorder>
                        </div>
                    </div>
                </div>
                <div className="">
                    <img
                        src="/robot.gif"
                        alt="Robot animation"
                        className=" max-w-[250px] sm:max-w-[300px] mx-auto"
                    />
                </div>
            </div>
        </>
    )
}